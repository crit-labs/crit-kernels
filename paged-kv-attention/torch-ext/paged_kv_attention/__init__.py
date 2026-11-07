"""Attention over paged KV caches in Triton: writes into pages, decode and speculative-verify windows split over KV
partitions merged by their log-sum-exp (with tree masks and block-sparse selection), and causal prompt chunks.

Pages. A cache layer is keys [pages][page][kv_heads][dim] and values [pages][page][kv_heads][value_dim]; position p of
a sequence lives at slot table[p // page] * page + p % page. Page kinds:

- "bf16", "fp16": 16-bit values;
- "fp8": E4M3 codes with one multiplier per layer for the keys and one for the values (value = code * multiplier);
- "fp8:row": E4M3 codes with an FP32 scale per position and KV head ([pages][page][kv_heads]): scale = max |x| / 448
  (at least 1e-30), code = x / scale rounded to the nearest E4M3 value, ties to even;
- "int8": INT8 codes with an FP32 scale per position and KV head: scale = max |x| / 127 (at least 1e-30), code = x / scale
  rounded to the nearest integer, ties to even.

Codes convert exactly to BF16 (or FP16) for the tensor cores; the keys' scale multiplies the scores and the values'
the probabilities. Softmax in FP32.

Windows (attend). A launch reads R rows; row r is a run of counts[r] <= columns tokens of one sequence (a decode token,
a verify window of a pending token and its drafts, a short prompt piece): tokens starts[r] .. of q, column c at position
firsts[r] + c, its keys and values written already, its page table tables[rows[r]]. Column c sees every position before
firsts[r] and the window's columns 0..c, or with `parents` (each token's parent column in its row, -1 for none: a draft
tree in topological order) the window columns among its ancestors and itself; with `blocks` ([tokens][k] KV block ids,
-1 for none, block_size positions each) only the positions of its listed blocks.

Keys are read in pieces of `piece` positions (piece_size); piece i belongs to partition i mod P (`partitions`), so
partition b walks pieces b, b + P, ... up to its row's last position: the work follows each row's length, read on the
device, not the cache's capacity, and one launch shape (a CUDA graph) serves every length. Each partition writes its
unnormalized output with its running maximum and sum (the partials: one row per column, rows' columns packed by
`offsets` or `columns` apart); a merge combines the partitions holding a column's keys in partition order.

A window launch's query tile (tile_for) holds the launch's columns of a KV head's query heads: up to 32 rows in two
warps, or 64 rows in four where the GPU multiplies four warps over 64 rows with mma.sync too (not sm_90's wgmma, not
sm_100's tcgen05: there such windows take 32-row tiles): on each GPU one instruction shape and one summation order for
every launch, and the row sums run on the tensor cores in a fixed order too. So a column's bits
depend on its own keys and query, the partitions, the piece and the page kind only: not on the rows beside it, not on
how many columns or rows its launch reads (one column decoded alone equals the same column inside a verify window).
A launch given another tile or warp count keeps that only where the GPU multiplies it with mma.sync too. The
partitions rule, the piece size, the tiles and their warps and stages are defaults measured on one sm_120 GPU (an RTX
5090). Past the first call of a shape nothing is
allocated but an `out` the caller does not give (the partials live in a workspace that grows outside a capture only,
or in the caller's), and nothing waits for the host.

Prompt chunks (prefill): one sequence's chunk of queries at positions position.. over its positions up to each query's
own, causal, through its page table; int8_qk quantizes the queries to INT8 per token and head for integer tensor
cores over INT8 pages (an approximation of the queries, twice the BF16 rate).
"""
import functools
import math

import torch
import triton

from . import _kernels

__all__ = ["KINDS", "PIECE", "MAX_COLUMNS", "attend", "partitions", "piece_size", "prefill", "tile_for",
           "workspace_floats", "write"]

KINDS = {"bf16": 0, "fp16": 0, "fp8": 1, "fp8:row": 2, "int8": 3}
PIECE = 32          # positions per piece of FP8 or INT8 pages at a 256-wide head (piece_size)
MAX_COLUMNS = 64    # columns of one row a launch reads (a tree's ancestor masks are 64-bit)

_workspaces = {}


def _check_kind(kind, key_scales):
    if kind not in KINDS:
        raise ValueError(f"page kinds are {', '.join(KINDS)}, not {kind!r}")
    if (KINDS[kind] >= 2) != (key_scales is not None):
        raise ValueError(f"{kind} pages {'take' if KINDS[kind] >= 2 else 'have no'} scales per position and head")
    return KINDS[kind]


@functools.lru_cache(maxsize=None)
def _capability(device):
    return torch.cuda.get_device_capability(device), torch.cuda.get_device_properties(device).multi_processor_count


def _hardware_fp8(device):
    return _capability(device)[0] >= (8, 9)


def _codes(pages, kind, device):
    """Pages as the kernels load them: FP8 pages as their bytes (converted by the GPU where it converts E4M3)."""
    if KINDS[kind] in (1, 2):
        return pages.view(torch.uint8)
    return pages


_dummies = {}


def _dummy(device):
    """A one-float tensor for the pointers a launch does not read (no scales, no tree, no lse)."""
    found = _dummies.get(device)
    if found is None:
        found = _dummies[device] = torch.zeros(1, dtype=torch.float32, device=device)
    return found


def partitions(device, kv_heads, kind):
    """A default partition count for a cache of `kv_heads` heads of page kind `kind` on `device`, for one long row:
    about one and a half programs per SM over converted pages (FP8, INT8), three quarters of one over 16-bit pages,
    whose pieces carry four times the bytes (a power of two within 8..128; measured best at 200K positions with 2, 4
    and 8 KV heads of 256 and 128 on an sm_120 GPU). A column's bits depend on it, so keep one count for every launch
    whose results must agree."""
    sms = _capability(torch.device(device))[1]
    programs = (3 * sms) // (2 if KINDS[kind] else 4)
    return max(8, min(128, triton.next_power_of_2(max(1, programs // max(1, kv_heads)))))


def piece_size(head_dim, value_dim, kind="bf16"):
    """Positions per piece for heads of these widths and a page kind: 16,384 head values for 16-bit pages, 8,192 for
    codes converted in registers (FP8, INT8), within 16..128 (measured best at 256- and 128-wide heads on an sm_120
    GPU). A column's bits depend on it; it follows the cache's geometry and page kind only."""
    values = 16384 if KINDS[kind] == 0 else 8192
    return max(16, min(128, values // max(head_dim, value_dim)))


def write(keys, values, slots, key_pages, value_pages, kind, key_scales=None, value_scales=None,
          multipliers=(1.0, 1.0)):
    """A step's keys [tokens][kv_heads][dim] and values [tokens][kv_heads][value_dim] (any float dtype) into their
    slots ([tokens] int64; a slot below 0 is skipped) of one layer's pages (any view whose storage is [slots][kv_heads]
    [dim]), encoded for the page kind (multipliers: an "fp8" layer's key and value multipliers, codes = x / multiplier
    clamped to +-448; the scales of "fp8:row" and "int8" pages are written beside their codes), in one launch.
    values and value_pages None: keys only (a cache of keys alone, an indexer's)."""
    k_kind = _check_kind(kind, key_scales)
    T, hkv, d = keys.shape
    if T == 0:
        return
    if keys.stride(-1) != 1:
        keys = keys.contiguous()
    heads = 2 * hkv
    if values is None:
        values, value_pages, value_scales, heads = keys, key_pages, key_scales, hkv
    elif values.stride(-1) != 1:
        values = values.contiguous()
    dv = values.shape[-1]
    kp, vp = key_pages, value_pages
    if k_kind in (1, 2):
        kp, vp = kp.view(torch.uint8), vp.view(torch.uint8)
    ks = key_scales if key_scales is not None else _dummy(keys.device)
    vs = value_scales if value_scales is not None else ks
    bd = triton.next_power_of_2(max(d, dv))
    _kernels._write_kernel[(T, heads)](keys, values, slots, kp, vp, ks, vs, keys.stride(0), keys.stride(1),
                                       values.stride(0), values.stride(1), float(multipliers[0]),
                                       float(multipliers[1]), HKV=hkv, D=d, DV=dv, BD=bd, KIND=k_kind,
                                       num_warps=1 if bd <= 128 else 2)


NARROW_ROWS = 32                 # rows of the tiles every GPU multiplies with mma.sync
WARPS, STAGES = 2, 3             # a tile of up to 32 rows
WIDE_ROWS, WIDE_WARPS, WIDE_STAGES = 64, 4, 4   # a 64-row tile, where four warps over it stay mma.sync
WARPGROUP_MMA = (9, 10, 11)      # major capabilities whose tensor cores take four warps over 64 rows as warpgroup MMA


def tile_for(columns, group, capability):
    """(rows, warps, stages) of a window launch's query tile, for `columns` columns over KV heads serving `group` query
    heads each on a GPU of `capability` (major, minor): the smallest power of two of 16 rows or more holding the
    columns' query rows, up to 32, in two warps; past 32 rows a 64-row tile in four warps (each piece's keys and values
    read once for its columns), except on GPUs whose tensor cores take four warps over 64 rows as warpgroup MMA (sm_90's
    wgmma, sm_100's and sm_110's tcgen05: another summation order), which read such windows in 32-row tiles; past 32
    heads a group, one column a tile of its power of two rows in two warps. So on each GPU every tile is multiplied with
    mma.sync, one instruction shape and summation order whatever the tile, and a column's bits are the same alone and
    in any window (the tests compile each choice for sm_80, sm_90, sm_100, sm_110 and sm_120 and check). Measured on
    one sm_120 GPU (an RTX 5090): 64 rows in four warps read windows of more than 32 query rows fastest there."""
    if group > NARROW_ROWS:
        return triton.next_power_of_2(group), WARPS, STAGES
    rows = columns * group
    if rows <= NARROW_ROWS or capability[0] in WARPGROUP_MMA:
        return max(16, triton.next_power_of_2(min(rows, NARROW_ROWS))), WARPS, STAGES
    return WIDE_ROWS, WIDE_WARPS, WIDE_STAGES


def workspace_floats(tokens, heads, partitions, value_dim):
    """FP32 floats of the partials a launch needs for `tokens` partial rows (attend's workspace): R x columns rows,
    or with offsets the rows' tokens packed."""
    return tokens * heads * partitions * (value_dim + 2)


def _workspace(device, tokens, heads, parts, dv, given=None):
    """Partials for tokens x heads x parts: outputs, maxima and sums, in `given` (one FP32 tensor of
    workspace_floats or more) or in a module workspace that grows outside a capture only."""
    need = tokens * heads * parts
    if given is not None:
        if given.dtype != torch.float32 or given.numel() < need * (dv + 2):
            raise ValueError(f"attend's workspace holds {given.numel()} floats; this launch needs {need * (dv + 2)} "
                             "(workspace_floats)")
        return given[:need * dv], given[need * dv:need * (dv + 1)], given[need * (dv + 1):need * (dv + 2)]
    found = _workspaces.get(device)
    if found is None or found[1].numel() < need:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("paged_kv_attention.attend: run a launch of this size once before capturing it (its "
                               "partials' workspace is allocated outside the capture)")
        size = max(need, 4096)
        found = (torch.empty(size * dv, dtype=torch.float32, device=device),
                 torch.empty(size, dtype=torch.float32, device=device),
                 torch.empty(size, dtype=torch.float32, device=device))
        _workspaces[device] = found
    po, pm, pl = found
    if po.numel() < need * dv:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("paged_kv_attention.attend: run a launch of this size once before capturing it")
        po = torch.empty(need * dv, dtype=torch.float32, device=device)
        _workspaces[device] = (po, pm, pl)
    return po, pm, pl


def attend(q, key_pages, value_pages, kind, tables, rows, starts, firsts, columns, page, *, counts=None,
           offsets=None, tokens=None, key_scales=None, value_scales=None, multipliers=(1.0, 1.0), scale=None,
           partitions=None, parents=None, blocks=None, block_size=None, out=None, lse=None, workspace=None, piece=None,
           tile=None, num_warps=None, num_stages=None, maxnreg=None):
    """Decode and verify windows over paged KV (see the module's description).

    q [tokens][heads][dim] BF16 or FP16 (FP16 for "fp16" pages); key_pages/value_pages one layer's pages, key_scales/
    value_scales their scales for "fp8:row" and "int8"; tables [S][width] int32 page tables; rows [R] the table row of
    each batch row; starts [R] int32 its first token in q; firsts [R] int64 its column 0's position; counts [R] int32
    its columns (None: `columns` each; more than `columns` read as `columns`); columns the widest row (<= MAX_COLUMNS),
    a Python int. offsets [R] int32 and tokens: each row's first partial row (its tokens' rank among the launch's,
    nondecreasing, rows' counts apart at least) and the partial rows the launch takes from offsets[0] on (at most
    R x columns), so the partials follow the launch's tokens; without them row r's partials are rows r x columns ...
    parents [tokens] (or a view of them) each token's parent column, -1 for none; blocks [tokens][k] int32 KV block ids
    and block_size. Writes out [tokens][heads][value_dim] (q's dtype; a new tensor each call when not given) at the
    rows' tokens, and lse [tokens][heads] FP32 (natural log of each row's softmax sum over its scaled scores) when
    given; returns out. workspace: an FP32 tensor of workspace_floats(partial rows, heads, partitions, value_dim) or
    more for the partials (default: one the module keeps per device, grown outside a capture). piece: positions per
    piece (default piece_size). tile, num_warps, num_stages, maxnreg: another launch configuration than tile_for()'s
    (a measurement's), whose columns keep their bits only where the GPU multiplies it with mma.sync too."""
    k_kind = _check_kind(kind, key_scales)
    T, hq, d = q.shape
    hkv, dv = key_pages.shape[-2], value_pages.shape[-1]
    R = int(starts.shape[0])
    if hq % hkv or not 1 <= columns <= MAX_COLUMNS or R == 0:
        raise ValueError("query heads a multiple of the KV heads, 1..64 columns a row, one row or more")
    if q.stride(-1) != 1:
        raise ValueError("q's last dimension is contiguous")
    if max(d, dv) > 4 * min(64, d, dv):
        raise ValueError("head dims are powers of two of 16 to 256 (keys and values within four 64-wide chunks)")
    if (offsets is None) != (tokens is None):
        raise ValueError("offsets come with the partial rows they take (tokens)")
    partial_rows = R * columns if offsets is None else int(tokens)
    g = hq // hkv
    device = q.device
    parts = int(partitions or globals()["partitions"](device, hkv, kind))
    scale = float(scale if scale is not None else d ** -0.5)
    po, pm, pl = _workspace(device, partial_rows, hq, parts, dv, workspace)
    out = torch.empty(T, hq, dv, dtype=q.dtype, device=device) if out is None else out
    kp, vp = _codes(key_pages, kind, device), _codes(value_pages, kind, device)
    ks = key_scales if key_scales is not None else _dummy(device)
    vs = value_scales if value_scales is not None else ks
    tree = parents is not None
    par = parents.reshape(-1) if tree else starts
    count_ptr = counts if counts is not None else starts
    off_ptr = offsets if offsets is not None else starts
    launch = {"maxnreg": int(maxnreg)} if maxnreg else {}
    piece = int(piece or piece_size(d, dv, kind))
    bm, warps, stages = tile_for(columns, g, _capability(device)[0])
    if tile:
        bm = max(16, triton.next_power_of_2(max(int(tile), g)))
    warps, stages = num_warps or warps, num_stages or stages
    common = dict(HQ=hq, HKV=hkv, G=g, D=d, DV=dv, PIECE=piece, PAGE=page, KIND=k_kind,
                  HW_FP8=_hardware_fp8(device), TREE=tree, COUNTS=counts is not None, OFFSETS=offsets is not None,
                  DC=min(64, d, dv))
    if blocks is None:
        cpt = bm // g
        grid = (triton.cdiv(columns, cpt), parts, R * hkv)
        _kernels._window_kernel[grid](q, kp, vp, ks, vs, tables, rows, starts, count_ptr, firsts, par, off_ptr, po, pm, pl,
                     q.stride(0), q.stride(1), tables.stride(0), tables.shape[1], parts, scale, float(multipliers[0]),
                     float(multipliers[1]), columns, COLS=columns, CPT=cpt, BLOCK_M=bm, num_warps=warps,
                     num_stages=stages, **common, **launch)
        fixed = 0
    else:
        if not block_size or block_size < 1:
            raise ValueError("block-sparse attention takes the blocks' size in positions")
        if blocks.stride(-1) != 1:
            blocks = blocks.contiguous()
        k = int(blocks.shape[-1])
        bm, warps = max(16, triton.next_power_of_2(g)), num_warps or WARPS   # one token's query heads a program
        stages = num_stages or STAGES
        grid = (parts, hkv, R * columns)
        _kernels._sparse_kernel[grid](q, kp, vp, ks, vs, tables, rows, starts, count_ptr, firsts, par, off_ptr,
                                      blocks, blocks.stride(0), k, int(block_size), po, pm, pl, q.stride(0),
                                      q.stride(1), tables.stride(0), tables.shape[1], parts, scale,
                                      float(multipliers[0]), float(multipliers[1]), columns, COLS=columns,
                                      BLOCK_M=bm, num_warps=warps, num_stages=stages, **common)
        fixed = triton.cdiv(k * int(block_size), piece)
    block_p = triton.next_power_of_2(parts)
    _kernels._merge_kernel[(columns, hq, R)](po, pm, pl, out, lse if lse is not None else pm, starts, count_ptr,
                                             firsts, off_ptr, parts, fixed, out.stride(0), out.stride(1), columns,
                                             COLS=columns, HQ=hq, DV=dv, PIECE=piece, COUNTS=counts is not None,
                                             OFFSETS=offsets is not None, LSE=lse is not None, BLOCK_P=block_p,
                                             CHUNK_P=min(16, block_p), num_warps=4)
    return out


def prefill(q, key_pages, value_pages, kind, table, position, page, *, key_scales=None, value_scales=None,
            multipliers=(1.0, 1.0), scale=None, out=None, lse=None, int8_qk=False):
    """A prompt chunk of one sequence: q [T][heads][dim] at positions position.. (their keys and values written),
    table [pages] int32 -> out [T][heads][dim] in q's dtype (and lse [T][heads] FP32 when given), causal. position: an
    int, or a one-element int64 device tensor read on the device (one launch for every position: a captured graph of
    the chunk replays at any; its keys reach position + T)."""
    k_kind = _check_kind(kind, key_scales)
    if int8_qk and k_kind != 3:
        raise ValueError("int8_qk multiplies INT8 queries with INT8 keys: int8 pages only")
    T, hq, d = q.shape
    hkv = key_pages.shape[-2]
    if value_pages.shape[-1] != d:
        raise ValueError("prompt chunks read keys and values of one width")
    device = q.device
    out = torch.empty_like(q) if out is None else out
    kp, vp = _codes(key_pages, kind, device), _codes(value_pages, kind, device)
    ks = key_scales if key_scales is not None else _dummy(device)
    vs = value_scales if value_scales is not None else ks
    block_n = 64 if d <= 128 else 32
    on_device = torch.is_tensor(position)
    _kernels._prefill_kernel[(triton.cdiv(T, 64), hq)](
        q, kp, vp, ks, vs, table, out, lse if lse is not None else ks, position if on_device else table, T,
        0 if on_device else int(position),
        float(scale if scale is not None else d ** -0.5), float(multipliers[0]), float(multipliers[1]), q.stride(0),
        q.stride(1), out.stride(0), out.stride(1), HQ=hq, HKV=hkv, G=hq // hkv, D=d, BLOCK_M=64, BLOCK_N=block_n,
        PAGE=page, KIND=k_kind, HW_FP8=_hardware_fp8(device), INT8_QK=int8_qk, LSE=lse is not None,
        POS_T=on_device, num_warps=4, num_stages=3 if int8_qk else 2)
    return out
