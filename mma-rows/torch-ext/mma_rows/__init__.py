"""Projections of 1 to 64 activation rows on tensor cores in one fixed tile, from crit's quantized-projections
technique: an NVFP4, FP8, MXFP8 or BF16 matrix [N][K] times BF16 rows [M][K], the activations used as they are
(weight only: W4A16, W8A16) or quantized as a prompt's are (FP8 per row: W8A8; NVFP4 per 16-value block: W4A4), FP32
sums. Every value enters mma.m16n8k16 exactly (BF16 operands, FP32 accumulate) and each weight is read once a call.
The order of every output's sum depends on N, K and `splits` only: a row gets the same bits alone, in a decode call of
up to 16 rows or in a prompt's tile of up to 64. plan() picks `groups`, `warps` (speed only) and `splits` (the sums'
order) from the matrix and the device, so a process keeps one plan per matrix.

Quantized activations go through a pre-pass into `scratch` (scratch_bytes); splits > 1 add their partial sums in
`partial` (partial_elements) with one int32 counter a row tile in `counters` (row_tiles(N, 1, warps): calls of more
than 32 rows take one group a warp; zeros, left zero). Calls that share those buffers run in stream order; nothing allocates and nothing waits for the host, so a CUDA graph captures a
call. The activations' quantizers alone (quantize_fp8_rows, quantize_nvfp4) give a provider's GEMM the same
activations (cuBLASLt's FP8 and NVFP4 block-scaled GEMMs for prompt rows).

Matrices as checkpoints store them: NVFP4 codes [N][K / 2] uint8 (element 2i in the low nibble of byte i), E4M3 block
scales [N][K / 16] row major or in cuBLASLt's 128 x 4 tiled layout (blocked), row_scale [N] FP32: one over each row's
global scale. FP8: E4M3 codes [N][K] and row_scale [N]. MXFP8: E4M3 codes [N][K] and E8M0 scales [N][K / 32]. BF16
[N][K]. K is a multiple of 32."""
import math

import torch

from ._ops import ops

WEIGHTS = {"nvfp4": 0, "fp8": 1, "mxfp8": 2, "bf16": 3}
ACTS = {"bf16": 0, "fp8_rows": 1, "nvfp4": 2}
MAX_ROWS = 64     # activation rows a call takes
STEP = 32         # K is a multiple of it
VALUES = {name: int(ops.step_values(code)) for name, code in WEIGHTS.items()}   # K a step takes (the kernel's)
MAX_WARPS, MAX_SPLITS = 8, 32
WARPS = 4          # warps a CTA, each its own 16 x groups matrix rows
CTAS_PER_SM = 3    # CTAs plan() asks of each multiprocessor, through splits of K where the rows are too few

__all__ = ["ACTS", "MAX_ROWS", "STEP", "VALUES", "WEIGHTS", "eligible", "mm_rows", "partial_elements", "plan",
           "quantize_fp8_rows", "quantize_nvfp4", "row_tiles", "scratch_bytes"]


def eligible(rows: int, columns: int) -> bool:
    """Whether mm_rows takes a matrix of this shape (K a multiple of STEP)."""
    return rows >= 1 and columns >= STEP and columns % STEP == 0


def plan(rows: int, columns: int, weights: str, device=None):
    """(groups, warps, splits) for a rows x columns matrix of `weights` on `device`, from the shape and the device's
    multiprocessor count alone: two 16-row groups a warp where the row tiles still give every multiprocessor one,
    WARPS warps a CTA, and the splits of K that give each multiprocessor CTAS_PER_SM CTAs (at most a step of K each)."""
    if weights not in WEIGHTS:
        raise ValueError(f"weights are {', '.join(WEIGHTS)}, not {weights!r}")
    if not eligible(rows, columns):
        raise ValueError(f"mm_rows takes K a multiple of {STEP}, not {columns}")
    sms = torch.cuda.get_device_properties(torch.device(device or "cuda")).multi_processor_count
    groups = 2 if math.ceil(rows / (32 * WARPS)) >= sms else 1
    tiles = row_tiles(rows, groups, WARPS)
    steps = math.ceil(columns / VALUES[weights])
    splits = max(1, min(MAX_SPLITS, steps, math.ceil(CTAS_PER_SM * sms / tiles)))
    return groups, WARPS, splits


def row_tiles(rows: int, groups: int, warps: int) -> int:
    return math.ceil(rows / (16 * groups * warps))


def scratch_bytes(tokens: int, columns: int) -> int:
    """Bytes of scratch quantized activations take for `tokens` rows of `columns`."""
    return (tokens * columns * 2 + 15) // 16 * 16 + tokens * 4


def partial_elements(tokens: int, rows: int, splits: int) -> int:
    return splits * tokens * rows if splits > 1 else 0


def mm_rows(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, weights: str, scales=None, row_scale=None,
            acts: str = "bf16", input_scale: float = 1.0, blocked: bool = False, groups: int = 1, warps: int = WARPS,
            splits: int = 1, scratch=None, partial=None, counters=None):
    """out [M][N] (FP32, BF16 or FP16) = x [M][K] (M 1..64, BF16, rows on 16-byte boundaries) times the matrix w.
    acts: bf16 (x as it is), fp8_rows (FP8 weights: each row in E4M3 by its own scale amax * fp32(1 / 448), at least
    1e-12; out = sum * row_scale * the row's scale), nvfp4 (NVFP4 weights: each 16-value block in E2M1 under
    input_scale, the block scale e4m3(min(amax * fp32(1 / 6) * input_scale, 448)); out = sum * row_scale *
    fp32(1 / input_scale)). groups, warps, splits: plan()'s."""
    ops.mm_rows(x, w, scales, row_scale, WEIGHTS[weights], ACTS[acts], float(input_scale), blocked, groups, warps,
                splits, scratch, partial, counters, out)
    return out


def quantize_fp8_rows(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor):
    """q [M][K] (float8_e4m3fn or uint8) and scale [M] FP32: BF16 rows in E4M3, each by its own scale, as mm_rows'
    fp8_rows activations (any M and K)."""
    ops.quantize_fp8_rows(x, q.view(torch.uint8), scale)
    return q, scale


def quantize_nvfp4(x: torch.Tensor, input_scale: float, codes: torch.Tensor, scales: torch.Tensor):
    """codes [M][K / 2] (uint8 or float4_e2m1fn_x2) and scales (E4M3 bytes, cuBLASLt's 128 x 4 tiled layout of
    [M][K / 16], rows padded to 128 and blocks to 4, the padding zero): BF16 rows in NVFP4 under input_scale, as
    mm_rows' nvfp4 activations (K a multiple of 16)."""
    ops.quantize_nvfp4(x, float(input_scale), codes.view(torch.uint8), scales.view(torch.uint8))
    return codes, scales
