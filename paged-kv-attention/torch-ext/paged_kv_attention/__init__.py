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

Keys are read in pieces of PIECE positions; piece i belongs to partition i mod P (`partitions`), so partition b walks
pieces b, b + P, ... up to its row's last position: the work follows each row's length, read on the device, not the
cache's capacity, and one launch shape (a CUDA graph) serves every length. Each partition writes its unnormalized
output with its running maximum and sum; a merge combines the partitions holding a column's keys in partition order.
A column's bits depend on its own keys and query, P and the page kind only: not on the rows beside it, not on how many
columns its launch reads (one column decoded alone equals the same column inside a verify window), not on the tile
shape (the row sums run on the tensor cores in a fixed order). Nothing is allocated past the first call of a shape
(the partials live in a workspace that grows outside a capture only), nothing waits for the host.

Prompt chunks (prefill): one sequence's chunk of queries at positions position.. over its positions up to each query's
own, causal, through its page table; int8_qk quantizes the queries to INT8 per token and head for integer tensor
cores over INT8 pages (an approximation of the queries, twice the BF16 rate).
"""
import functools
import math

import torch
import triton

from . import _kernels

__all__ = ["KINDS", "PIECE", "MAX_COLUMNS", "attend", "partitions", "piece_size", "prefill", "workspace_floats",
           "write"]

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


def partitions(device, kv_heads):
    """A default partition count for a cache of `kv_heads` heads on `device`: about one and a half programs per SM
    for one row (a power of two within 8..128; measured best at 256-wide heads with 4 KV heads and 128-wide with 8 on
    an sm_120 GPU). A column's bits depend on it, so keep one count for every launch whose results must agree."""
    sms = _capability(torch.device(device))[1]
    return max(8, min(128, triton.next_power_of_2(max(1, (3 * sms) // (2 * max(1, kv_heads))))))


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


def _tile(columns, group, dim, rows=None):
    """(rows of a query tile, columns per tile): a tile holds whole columns of one KV head's query heads; `rows` asks
    for a tile of that many rows (a power of two of 16 or more) where it holds a column."""
    cap = 64 if dim > 128 else 128
    if group >= cap:
        return triton.next_power_of_2(group), 1
    if rows is not None:
        rows = max(16, triton.next_power_of_2(min(max(int(rows), group), cap)))
    else:
        rows = max(16, triton.next_power_of_2(min(columns * group, cap)))
    return rows, rows // group


def workspace_floats(rows, columns, heads, partitions, value_dim):
    """FP32 floats of the partials a launch of `rows` rows of `columns` columns needs (attend's workspace)."""
    return rows * columns * heads * partitions * (value_dim + 2)


def _launch(rows, num_warps, num_stages):
    """Warps and pipeline stages of a query tile of `rows` rows: a warp per 16 rows, two at least (two warps with three
    stages read a one-column tile fastest, four with four a 64-row one, measured at a 256-wide head on an sm_120 GPU).
    Neither changes a column's bits (every per-row sum runs on the tensor cores in a fixed order)."""
    warps = num_warps or max(2, rows // 16)
    return warps, num_stages or (3 if warps <= 2 else 4)


def _workspace(device, rows, columns, heads, parts, dv, given=None):
    """Partials for rows x columns x heads x parts: outputs, maxima and sums, in `given` (one FP32 tensor of
    workspace_floats or more) or in a module workspace that grows outside a capture only."""
    need = rows * columns * heads * parts
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
           key_scales=None, value_scales=None, multipliers=(1.0, 1.0), scale=None, partitions=None, parents=None,
           blocks=None, block_size=None, out=None, lse=None, tile=None, workspace=None, piece=None, num_warps=None,
           num_stages=None, maxnreg=None):
    """Decode and verify windows over paged KV (see the module's description).

    q [tokens][heads][dim] BF16 or FP16 (FP16 for "fp16" pages); key_pages/value_pages one layer's pages, key_scales/
    value_scales their scales for "fp8:row" and "int8"; tables [S][width] int32 page tables; rows [R] the table row of
    each batch row; starts [R] int32 its first token in q; firsts [R] int64 its column 0's position; counts [R] int32
    its columns (None: `columns` each); columns the widest row (<= MAX_COLUMNS), a Python int. parents [tokens] (or a
    view of them) each token's parent column, -1 for none; blocks [tokens][k] int32 KV block ids and block_size.
    Writes out [tokens][heads][value_dim] (q's dtype) at the rows' tokens, and lse [tokens][heads] FP32 (natural log
    of each row's softmax sum over its scaled scores) when given; returns out. tile: the query tile's rows (default:
    the launch's columns of query heads, up to 64 or 128); a column's bits do not depend on it. workspace: an FP32
    tensor of workspace_floats(R, columns, heads, partitions, value_dim) or more for the partials (default: one the
    module keeps per device, grown outside a capture)."""
    k_kind = _check_kind(kind, key_scales)
    T, hq, d = q.shape
    hkv, dv = key_pages.shape[-2], value_pages.shape[-1]
    R = int(starts.shape[0])
    if hq % hkv or not 1 <= columns <= MAX_COLUMNS or R == 0:
        raise ValueError("query heads a multiple of the KV heads, 1..64 columns a row, one row or more")
    if q.stride(-1) != 1:
        raise ValueError("q's last dimension is contiguous")
    g = hq // hkv
    device = q.device
    parts = int(partitions or globals()["partitions"](device, hkv))
    scale = float(scale if scale is not None else d ** -0.5)
    out = torch.empty(T, hq, dv, dtype=q.dtype, device=device) if out is None else out
    kp, vp = _codes(key_pages, kind, device), _codes(value_pages, kind, device)
    ks = key_scales if key_scales is not None else _dummy(device)
    vs = value_scales if value_scales is not None else ks
    tree = parents is not None
    par = parents.reshape(-1) if tree else starts
    count_ptr = counts if counts is not None else starts
    po, pm, pl = _workspace(device, R, columns, hq, parts, dv, workspace)
    if max(d, dv) > 4 * min(64, d, dv):
        raise ValueError("head dims are powers of two of 16 to 256 (keys and values within four 64-wide chunks)")
    launch = {"maxnreg": int(maxnreg)} if maxnreg else {}
    piece = int(piece or piece_size(d, dv, kind))
    common = dict(HQ=hq, HKV=hkv, G=g, D=d, DV=dv, PIECE=piece, PAGE=page, KIND=k_kind,
                  HW_FP8=_hardware_fp8(device), TREE=tree, COUNTS=counts is not None, DC=min(64, d, dv))
    if blocks is None:
        bm, cpt = _tile(columns, g, max(d, dv), tile)
        warps, stages = _launch(bm, num_warps, num_stages)
        grid = (triton.cdiv(columns, cpt), parts, R * hkv)
        _kernels._window_kernel[grid](q, kp, vp, ks, vs, tables, rows, starts, count_ptr, firsts, par, po, pm, pl,
                                      q.stride(0), q.stride(1), tables.stride(0), tables.shape[1], parts, scale,
                                      float(multipliers[0]), float(multipliers[1]), columns, COLS=columns, CPT=cpt,
                                      BLOCK_M=bm, num_warps=warps, num_stages=stages, **common, **launch)
        fixed = 0
    else:
        if not block_size or block_size < 1:
            raise ValueError("block-sparse attention takes the blocks' size in positions")
        if blocks.stride(-1) != 1:
            blocks = blocks.contiguous()
        k = int(blocks.shape[-1])
        bm = max(16, triton.next_power_of_2(g))
        warps, stages = _launch(bm, num_warps, num_stages)
        grid = (parts, hkv, R * columns)
        _kernels._sparse_kernel[grid](q, kp, vp, ks, vs, tables, rows, starts, count_ptr, firsts, par, blocks,
                                      blocks.stride(0), k, int(block_size), po, pm, pl, q.stride(0), q.stride(1),
                                      tables.stride(0), tables.shape[1], parts, scale, float(multipliers[0]),
                                      float(multipliers[1]), columns, COLS=columns, BLOCK_M=bm, num_warps=warps,
                                      num_stages=stages, **common)
        fixed = triton.cdiv(k * int(block_size), piece)
    block_p = triton.next_power_of_2(parts)
    _kernels._merge_kernel[(columns, hq, R)](po, pm, pl, out, lse if lse is not None else pm, starts, count_ptr,
                                             firsts, parts, fixed, out.stride(0), out.stride(1), columns, COLS=columns,
                                             HQ=hq, DV=dv, PIECE=piece, COUNTS=counts is not None,
                                             LSE=lse is not None, BLOCK_P=block_p, CHUNK_P=min(16, block_p),
                                             num_warps=4)
    return out


def prefill(q, key_pages, value_pages, kind, table, position, page, *, key_scales=None, value_scales=None,
            multipliers=(1.0, 1.0), scale=None, out=None, lse=None, int8_qk=False):
    """A prompt chunk of one sequence: q [T][heads][dim] at positions position.. (their keys and values written),
    table [pages] int32 -> out [T][heads][dim] in q's dtype (and lse [T][heads] FP32 when given), causal."""
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
    _kernels._prefill_kernel[(triton.cdiv(T, 64), hq)](
        q, kp, vp, ks, vs, table, out, lse if lse is not None else ks, T, int(position),
        float(scale if scale is not None else d ** -0.5), float(multipliers[0]), float(multipliers[1]), q.stride(0),
        q.stride(1), out.stride(0), out.stride(1), HQ=hq, HKV=hkv, G=hq // hkv, D=d, BLOCK_M=64, BLOCK_N=block_n,
        PAGE=page, KIND=k_kind, HW_FP8=_hardware_fp8(device), INT8_QK=int8_qk, LSE=lse is not None, num_warps=4,
        num_stages=3 if int8_qk else 2)
    return out
