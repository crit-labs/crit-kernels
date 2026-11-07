"""Normalization, rotary positions and gated products around a transformer's matmuls, one Triton kernel per call:
RMSNorm with a residual added in place (an FP32 residual stream beside 16-bit rows), an offset weight (a zero-centred
weight stored as weight - 1), one weight per group of rows (several residual streams normalized at once) and a chosen
output dtype; query/key head norms, alone or with rotary positions; the gated RMSNorm of linear-attention outputs;
rotary positions (half or interleaved pairs, partial, inverse frequencies or a cos|sin table); SiLU, GELU and sigmoid
products; L2 norms; and the mixes of several residual streams: a gated or weighted sum over the streams, and one row
added into every stream with a weight per stream.

Inputs BF16, FP16 or FP32 on a CUDA device, each read in its own dtype; FP32 inside; one rounding, to out_dtype (by
default the dtype of the tensor normalized or gated). The last dimension is contiguous and the others may have any
strides (views of a fused projection's output). Each row's bits are independent of the rows beside it and of the row
count, so decode rows, verify windows and prompt rows agree: a launch's specialization never depends on the row count
(dimensions of size 1 keep their strides), and a layout the kernels cannot read aligned (a base address or a row stride
that is not a multiple of 16) is copied into aligned buffers first, so every row is read by one layout. Nothing
synchronizes with the host and every allocation (the result unless `out` is given, and those copies) comes from
torch's caching allocator, so CUDA graphs capture every call once its shape has been compiled (call each shape once
before capturing it). Every tensor of a call is on one device, the one the call runs on.
"""
import contextlib
from typing import Optional

import torch

from . import _triton

__all__ = ["gated_rms_norm", "gelu_mul", "l2_norm", "rms_norm", "rms_norm_heads", "rms_norm_rope", "rope",
           "sigmoid_mul", "silu_mul", "stream_add", "stream_sum"]

FLOATS = (torch.bfloat16, torch.float16, torch.float32)
ROW_ELEMENTS = 4096      # values a program holds: rows of up to this width one a program, shorter ones several
WIDEST_BLOCK = 16384     # rows wider than this run in two passes of CHUNK values
CHUNK = 4096
PRODUCT_BLOCK = 1024
STYLES = {"half": 1, "interleaved": 2}
GATES = {"silu": 1, "sigmoid": 2}
LAYOUTS = ("halves", "interleaved")
APPROXIMATIONS = {"none": 1, "tanh": 2}
STREAM_GATES = {"none": 4, "silu": 0, "gelu": 1, "gelu_tanh": 2, "sigmoid": 3}

_CONFIG = {}


def _cdiv(a, b):
    return -(-a // b)


def _config(width):
    """(BD, HB, warps): a program's block of HB rows of BD values and its warp count, from the row width alone."""
    found = _CONFIG.get(width)
    if found is None:
        block = max(16, 1 << (width - 1).bit_length())
        rows = max(1, ROW_ELEMENTS // block)
        found = _CONFIG[width] = (block, rows, max(1, min(16, block * rows // 512)))
    return found


def _check(x, what="x", device=None):
    if not x.is_cuda or x.dtype not in FLOATS:
        raise ValueError(f"{what} is a BF16, FP16 or FP32 CUDA tensor, not {x.dtype} on {x.device}")
    if device is not None and x.device != device:
        raise ValueError(f"{what} is on {x.device}, the call's tensors on {device}")
    if x.dim() < 1 or x.shape[-1] < 1 or (x.stride(-1) != 1 and x.shape[-1] > 1):
        raise ValueError(f"{what}'s last dimension is contiguous")


def _on(x):
    """The launch context: x's device when it is not the current one."""
    if x.device.index == torch.cuda.current_device():
        return contextlib.nullcontext()
    return torch.cuda.device(x.device)


def _aligned(tensors, strides):
    """Whether the kernels read every tensor aligned: 16-byte base addresses and row strides that are multiples of 16
    (a launch's specialization then never depends on the rows given)."""
    return (all(t.data_ptr() % 16 == 0 for t in tensors)
            and all(a % 16 == 0 and b % 16 == 0 for a, b in strides))


def _padded(t, copy=True):
    """t's values in an aligned buffer whose rows are padded to a multiple of 16 values (a view of t's shape)."""
    width = t.shape[-1]
    buffer = torch.empty((*t.shape[:-1], _cdiv(width, 16) * 16), dtype=t.dtype, device=t.device)[..., :width]
    if copy:
        buffer.copy_(t)
    return buffer


def _collapse(shape, tensors):
    """(A, B, [(sa, sb)]) such that row a * B + b of every tensor sits at a * sa + b * sb, from shape[:-1] and each
    tensor's strides, or None when the leading dimensions do not merge into two levels for every tensor. Dimensions of
    size 1 count with their strides like any other, so a call of one row merges the way a call of many does."""
    dims = []
    for i, size in enumerate(shape[:-1]):
        strides = [t.stride(i) for t in tensors]
        if dims and all(outer == size * inner for outer, inner in zip(dims[-1][1], strides)):
            dims[-1] = (dims[-1][0] * size, strides)
        else:
            dims.append((size, strides))
    if len(dims) > 2:
        return None
    if not dims:
        return 1, 1, [(0, 0)] * len(tensors)
    if len(dims) == 1:
        return dims[0][0], 1, [(s, 0) for s in dims[0][1]]
    (a, outer), (b, inner) = dims
    return a, b, list(zip(outer, inner))


def _result(x, shape, out_dtype, out):
    out_dtype = out_dtype or x.dtype
    if out_dtype not in FLOATS:
        raise ValueError(f"out_dtype is bf16, fp16 or fp32, not {out_dtype}")
    if out is None:
        return torch.empty(shape, dtype=out_dtype, device=x.device)
    if tuple(out.shape) != tuple(shape) or out.dtype != out_dtype or out.device != x.device:
        raise ValueError(f"out is {tuple(out.shape)} {out.dtype} on {out.device}; the result is {tuple(shape)} "
                         f"{out_dtype} on {x.device}")
    if shape[-1] > 1 and out.stride(-1) != 1:
        raise ValueError("out's last dimension is contiguous")
    return out


def _weight(weight, x):
    """(groups, row stride, grouped) of a weight [dim], or [groups][dim] whose row g weighs the rows of x at index g of
    its second-to-last dimension."""
    width = x.shape[-1]
    if weight is None:
        return 1, 0, False
    if weight.device != x.device or weight.shape[-1] != width or weight.stride(-1) != 1:
        raise ValueError(f"a weight of shape {tuple(weight.shape)} for rows of {width} (its last dimension contiguous, "
                         f"on {x.device})")
    if weight.dim() == 1:
        return 1, 0, False
    if weight.dim() != 2 or x.dim() < 2 or x.shape[-2] != weight.shape[0]:
        raise ValueError(f"a weight of shape {tuple(weight.shape)}: [dim], or [groups][dim] with groups x's "
                         f"second-to-last dimension (x is {tuple(x.shape)})")
    return weight.shape[0], weight.stride(0), True


def _rows(x, weight, eps, offset, norm, residual=None, z=None, gate=0, out_dtype=None, out=None):
    """The rows kernel without rope (or the two-pass kernel past WIDEST_BLOCK) over every last-dimension row."""
    _check(x)
    width = x.shape[-1]
    groups, swg, grouped = _weight(weight, x)
    for other, what in ((residual, "residual"), (z, "z")):
        if other is not None:
            _check(other, what, x.device)
            if other.shape != x.shape:
                raise ValueError(f"{what} of shape {tuple(other.shape)} for x of shape {tuple(x.shape)}")
    y = _result(residual if residual is not None else x, x.shape, out_dtype, out)
    present = [t for t in (x, residual, z, y) if t is not None]
    layout = _collapse(x.shape, present)
    if layout is None or not _aligned(present + ([weight] if weight is not None else []), layout[2]):
        # a layout the kernels cannot read aligned in two levels: aligned padded copies, the results copied back
        kept = None if residual is None else _padded(residual)
        found = _padded(y, copy=False)
        if weight is not None and weight.data_ptr() % 16:
            weight = weight.clone()
        _rows(_padded(x), weight, eps, offset, norm, kept, None if z is None else _padded(z), gate, y.dtype, found)
        if kept is not None:
            residual.copy_(kept)
        y.copy_(found)
        return y
    A, B, strides = layout
    found = iter(strides)
    sx = next(found)
    sr = next(found) if residual is not None else (0, 0)
    sz = next(found) if z is not None else (0, 0)
    sy = next(found)
    n_rows = A * B
    if n_rows == 0:
        return y
    block, rows, warps = _config(width)
    w = weight if weight is not None else x
    r = residual if residual is not None else x
    g = z if z is not None else x
    with _on(x):
        if block <= WIDEST_BLOCK:
            _triton.rows_kernel[(_cdiv(n_rows, rows),)](
                x, *sx, r, *sr, g, *sz, w, groups, swg, y, *sy, x, x, 0, 1, n_rows, B, 1, width, 0, float(eps),
                float(offset), 1.0, NORM=norm, HAS_RES=residual is not None, HAS_W=weight is not None,
                GROUPED=grouped, GATE=gate, ROPE=0, TABLE=False, IN_PLACE=False, HB=rows, BD=block, num_warps=warps)
        else:
            _triton.long_rows_kernel[(n_rows,)](
                x, *sx, r, *sr, g, *sz, w, groups, swg, y, *sy, B, width, float(eps), float(offset),
                NORM=norm, HAS_RES=residual is not None, HAS_W=weight is not None, GATE=gate, CHUNK=CHUNK, num_warps=8)
    return y


def rms_norm(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float, offset: float = 0.0,
             residual: Optional[torch.Tensor] = None, out_dtype: Optional[torch.dtype] = None,
             out: Optional[torch.Tensor] = None):
    """y = x / sqrt(mean(x^2) + eps) * (offset + weight) per last-dimension row: weight [dim], [groups][dim] (row g
    for the rows at index g of x's second-to-last dimension: several streams or heads, each its own weight) or None.
    With residual (x's shape, its own dtype), the residual becomes residual + x in place, rounded to its dtype, those
    rows are normalized, and (y, residual) is returned; y's dtype defaults to the residual's then."""
    y = _rows(x, weight, eps, offset, 1, residual=residual, out_dtype=out_dtype, out=out)
    return y if residual is None else (y, residual)


def rms_norm_heads(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float, offset: float = 0.0,
                   out_dtype: Optional[torch.dtype] = None, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """rms_norm of each head of x [tokens][heads][dim] (query and key norms), weight [dim] or [heads][dim]."""
    return _rows(x, weight, eps, offset, 1, out_dtype=out_dtype, out=out)


def gated_rms_norm(x: torch.Tensor, z: torch.Tensor, weight: Optional[torch.Tensor], eps: float, offset: float = 0.0,
                   activation: str = "silu", out_dtype: Optional[torch.dtype] = None,
                   out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """rms_norm of each last-dimension row of x times activation(z) element by element (silu or sigmoid)."""
    if activation not in GATES:
        raise ValueError(f"a gated norm's activation is one of {', '.join(GATES)}, not {activation!r}")
    return _rows(x, weight, eps, offset, 1, z=z, gate=GATES[activation], out_dtype=out_dtype, out=out)


def l2_norm(x: torch.Tensor, eps: float = 1e-6, out_dtype: Optional[torch.dtype] = None,
            out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x / sqrt(sum(x^2) + eps) per last-dimension row."""
    return _rows(x, None, eps, 0.0, 2, out_dtype=out_dtype, out=out)


def _rotary(freqs, rotary_dim, width, device):
    if freqs.device != device or freqs.dtype != torch.float32 or freqs.stride(-1) != 1:
        raise ValueError(f"freqs is a contiguous FP32 tensor on {device}")
    if freqs.dim() == 1:
        found = 2 * freqs.shape[0]
    elif freqs.dim() == 2:
        found = freqs.shape[1]
    else:
        raise ValueError("freqs is [rotary_dim / 2] inverse frequencies or a [positions][rotary_dim] cos|sin table")
    rotary_dim = found if rotary_dim is None else int(rotary_dim)
    if rotary_dim != found or rotary_dim < 2 or rotary_dim % 2 or rotary_dim > width:
        raise ValueError(f"rotary_dim {rotary_dim}: even, at most the heads' {width} values, and what freqs give "
                         f"({found})")
    return rotary_dim


def _rope(x, positions, freqs, rotary_dim, style, scale, weight, eps, offset, norm, out_dtype, out):
    _check(x)
    if x.dim() not in (2, 3):
        raise ValueError(f"rope rotates [tokens][heads][dim] or [tokens][dim], not {tuple(x.shape)}")
    if style not in STYLES:
        raise ValueError(f"style is one of {', '.join(STYLES)}, not {style!r}")
    tokens, width = x.shape[0], x.shape[-1]
    heads = x.shape[1] if x.dim() == 3 else 1
    if (positions.shape != (tokens,) or positions.device != x.device
            or positions.dtype not in (torch.int64, torch.int32)):
        raise ValueError(f"positions [tokens] int64 or int32 on {x.device}, not {tuple(positions.shape)} "
                         f"{positions.dtype} on {positions.device}")
    groups, swg, grouped = _weight(weight, x if x.dim() == 3 else x[:, None, :])
    rotary_dim = _rotary(freqs, rotary_dim, width, x.device)
    block, rows, warps = _config(width)
    if block > WIDEST_BLOCK:
        raise ValueError(f"heads of {width} values are wider than rope's {WIDEST_BLOCK}")
    rows = min(rows, 1 << (heads - 1).bit_length())
    warps = max(1, min(16, block * rows // 512))
    y = _result(x, x.shape, out_dtype, out)
    if tokens == 0 or heads == 0:
        return y
    positions = positions.contiguous()
    sx = (x.stride(0), x.stride(1) if x.dim() == 3 else 0)
    sy = (y.stride(0), y.stride(1) if y.dim() == 3 else 0)
    if not _aligned([x, y] + ([weight] if weight is not None else []), [sx, sy]):
        # a layout the kernel cannot read aligned: an aligned padded copy, the result copied back (in place too)
        found = _padded(y, copy=False)
        if weight is not None and weight.data_ptr() % 16:
            weight = weight.clone()
        _rope(_padded(x), positions, freqs, rotary_dim, style, scale, weight, eps, offset, norm, y.dtype, found)
        y.copy_(found)
        return y
    n_blocks = _cdiv(heads, rows)
    table = freqs.dim() == 2
    with _on(x):
        _triton.rows_kernel[(tokens * n_blocks,)](
            x, *sx, x, 0, 0, x, 0, 0, weight if weight is not None else x, groups, swg, y, *sy,
            positions, freqs, freqs.stride(0) if table else 0, freqs.shape[0] if table else 1, tokens * heads, heads,
            n_blocks, width, rotary_dim, float(eps), float(offset), float(scale),
            NORM=norm, HAS_RES=False, HAS_W=weight is not None, GROUPED=grouped, GATE=0, ROPE=STYLES[style],
            TABLE=table, IN_PLACE=y.data_ptr() == x.data_ptr(), HB=rows, BD=block, num_warps=warps)
    return y


def rope(x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor, rotary_dim: Optional[int] = None,
         style: str = "half", scale: float = 1.0, out_dtype: Optional[torch.dtype] = None,
         out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Each token's heads (x [tokens][heads][dim] or [tokens][dim]) rotated at its position (positions [tokens]) on
    their first rotary_dim values, the rest passed through. freqs: [rotary_dim / 2] FP32 inverse frequencies (angle
    fp32(position) * freq in FP32) or a [positions][rotary_dim] FP32 table (cos of every pair, then sin; positions
    outside it clamped to it). style half (pairs j, j + rotary_dim / 2) or interleaved (2j, 2j + 1); scale multiplies
    cos and sin; out may be x (in place)."""
    return _rope(x, positions, freqs, rotary_dim, style, scale, None, 0.0, 0.0, 0, out_dtype, out)


def rms_norm_rope(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float, positions: torch.Tensor,
                  freqs: torch.Tensor, offset: float = 0.0, rotary_dim: Optional[int] = None, style: str = "half",
                  scale: float = 1.0, out_dtype: Optional[torch.dtype] = None,
                  out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """rms_norm_heads then rope, in FP32 between them (one rounding, to out_dtype)."""
    return _rope(x, positions, freqs, rotary_dim, style, scale, weight, eps, offset, 1, out_dtype, out)


def _product(gate, up, activation, out_dtype, out, interleaved=False, source=None):
    """y = activation(gate) * up over every last-dimension row; interleaved: both alternate in `source`."""
    _check(gate if source is None else source)
    if up is not None:
        _check(up, "up", (gate if source is None else source).device)
    width = gate.shape[-1]
    y = _result(source if source is not None else gate, gate.shape, out_dtype, out)
    present = [source] if interleaved else [gate, up]
    layout = _collapse(gate.shape if not interleaved else source.shape, present + [y])
    if layout is None:
        if interleaved:
            source = source.contiguous()
            gate = source[..., 0::2]
        else:
            gate, up = gate.contiguous(), up.contiguous()
        target = y if _collapse(y.shape, [y]) is not None else torch.empty_like(y)
        _product(gate, up, activation, y.dtype, target, interleaved, source)
        if target is not y:
            y.copy_(target)
        return y
    A, B, strides = layout
    n_rows = A * B
    if n_rows == 0 or width == 0:
        return y
    sg = strides[0]
    su = strides[1] if not interleaved else (0, 0)
    sy = strides[-1]
    n_blocks = _cdiv(width, PRODUCT_BLOCK)
    first = source if interleaved else gate
    with _on(first):
        _triton.act_mul_kernel[(n_rows * n_blocks,)](
            first, *sg, first if interleaved else up, *su, y, *sy, n_rows, B, width, n_blocks,
            ACT=activation, INTERLEAVED=interleaved, BLOCK=PRODUCT_BLOCK, num_warps=4)
    return y


def _split(x, up, layout, activation, out_dtype, out):
    if up is not None:
        if up.shape != x.shape:
            raise ValueError(f"up of shape {tuple(up.shape)} for gates of shape {tuple(x.shape)}")
        return _product(x, up, activation, out_dtype, out)
    if x.shape[-1] % 2:
        raise ValueError(f"gate and up halves of an even width, not {x.shape[-1]}")
    half = x.shape[-1] // 2
    if layout == "halves":
        return _product(x[..., :half], x[..., half:], activation, out_dtype or x.dtype, out)
    if layout == "interleaved":
        _check(x)
        return _product(x[..., 0::2], None, activation, out_dtype or x.dtype, out, interleaved=True, source=x)
    raise ValueError(f"layout is one of {', '.join(LAYOUTS)}, not {layout!r}")


def silu_mul(x: torch.Tensor, up: Optional[torch.Tensor] = None, layout: str = "halves",
             out_dtype: Optional[torch.dtype] = None, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """silu(gate) * up element by element: x the gates and up the values, or x [..., 2n] holding both (layout halves:
    [gate | up]; interleaved: gate at even, up at odd columns), y [..., n]."""
    return _split(x, up, layout, 0, out_dtype, out)


def gelu_mul(x: torch.Tensor, up: Optional[torch.Tensor] = None, layout: str = "halves", approximate: str = "none",
             out_dtype: Optional[torch.dtype] = None, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """gelu(gate) * up as silu_mul; approximate none (erf) or tanh."""
    if approximate not in APPROXIMATIONS:
        raise ValueError(f"approximate is one of {', '.join(APPROXIMATIONS)}, not {approximate!r}")
    return _split(x, up, layout, APPROXIMATIONS[approximate], out_dtype, out)


def sigmoid_mul(x: torch.Tensor, gate: torch.Tensor, out_dtype: Optional[torch.dtype] = None,
                out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x * sigmoid(gate) element by element (an attention output's gate)."""
    if gate.shape != x.shape:
        raise ValueError(f"a gate of shape {tuple(gate.shape)} for x of shape {tuple(x.shape)}")
    _check(x)
    return _product(gate, x, 3, out_dtype or x.dtype, out)


def _streams(x, what):
    _check(x, what)
    if x.dim() != 3:
        raise ValueError(f"{what} is [tokens][streams][dim], not {tuple(x.shape)}")


def stream_sum(x: torch.Tensor, gate: Optional[torch.Tensor] = None, activation: str = "none", scale: float = 1.0,
               out_dtype: Optional[torch.dtype] = None, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """y[t] = scale * the sum over streams s, in stream order, of activation(gate[t][s]) * x[t][s]: x [tokens][streams]
    [dim]; gate x's shape (one gate per value) or [tokens][streams] (one per stream), or None for the plain sum
    (scale 1 / streams: their mean); activation none (the gate itself), silu, gelu, gelu_tanh or sigmoid. y
    [tokens][dim]. A per-token mix of S streams into S streams is S calls, each into its output stream's out= view."""
    _streams(x, "x")
    tokens, streams, width = x.shape
    if activation not in STREAM_GATES:
        raise ValueError(f"activation is one of {', '.join(STREAM_GATES)}, not {activation!r}")
    if gate is not None:
        if gate.device != x.device or gate.dtype not in FLOATS or tuple(gate.shape) not in ((tokens, streams, width),
                                                                                          (tokens, streams)):
            raise ValueError(f"a gate of shape {tuple(gate.shape)} for streams of shape {tuple(x.shape)}: one per "
                             "value or one per stream")
        per_value = gate.dim() == 3
        if per_value and gate.stride(-1) != 1 and width > 1:
            gate = gate.contiguous()
        sg = (gate.stride(0), gate.stride(1), 1 if per_value else 0)
    else:
        sg = (0, 0, 0)
    y = _result(x, (tokens, width), out_dtype, out)
    if tokens == 0 or streams == 0:
        return y.zero_()
    n_blocks = _cdiv(width, PRODUCT_BLOCK)
    with _on(x):
        _triton.stream_sum_kernel[(tokens * n_blocks,)](
            x, x.stride(0), x.stride(1), gate if gate is not None else x, *sg, y, y.stride(0), streams, width,
            n_blocks, float(scale), HAS_GATE=gate is not None, ACT=STREAM_GATES[activation], BLOCK=PRODUCT_BLOCK,
            num_warps=4)
    return y


def stream_add(streams: torch.Tensor, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """streams[t][s] += weights[t][s] * x[t] for every stream s, in place (weights [tokens][streams], or None for 1):
    one row added into every residual stream; streams [tokens][streams][dim] in their own dtype, x [tokens][dim]."""
    _streams(streams, "streams")
    _check(x, "x", streams.device)
    tokens, count, width = streams.shape
    if tuple(x.shape) != (tokens, width):
        raise ValueError(f"x of shape {tuple(x.shape)} for streams of shape {tuple(streams.shape)}")
    if weights is not None and (weights.device != streams.device or tuple(weights.shape) != (tokens, count)):
        raise ValueError(f"weights of shape {tuple(weights.shape)} on {weights.device}: one per token and stream, on "
                         f"{streams.device}")
    if tokens == 0:
        return streams
    n_blocks = _cdiv(width, PRODUCT_BLOCK)
    sw = (weights.stride(0), weights.stride(1)) if weights is not None else (0, 0)
    with _on(streams):
        _triton.stream_add_kernel[(tokens * n_blocks,)](
            streams, streams.stride(0), streams.stride(1), x, x.stride(0), weights if weights is not None else x,
            *sw, count, width, n_blocks, HAS_W=weights is not None, BLOCK=PRODUCT_BLOCK, num_warps=4)
    return streams
