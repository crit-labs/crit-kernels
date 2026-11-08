"""Decode GEMVs from crit's quantized-projections technique, weight only: NVFP4 weights (W4A16), FP8 E4M3 weights with a
row scale or MXFP8's E8M0 scale per 32 inputs (W8A16) and BF16 weights times 1 to 16 rows of 16-bit activations on
CUDA cores; an NVFP4 GEMV for 1 to 8 rows in Triton; an FP8 W8A16 GEMV for 1 to 16 rows on tensor cores in Triton
(gemv_fp8_dot: a program of 16 rows by a block of the matrix, tl.dot, a fixed split of K added in order; fp8_dot_tile
gives a matrix its program shape); and the decoders that write an NVFP4 matrix's values for a provider's GEMM. FP32
sums; each row's result is the same bits alone and beside the other rows of a call (the order never depends on the row
count). The GEMVs sum in different orders, so a process keeps to one a matrix.

NVFP4 matrices as checkpoints store them: codes [N][K / 2] uint8 (element 2i in the low nibble of byte i), E4M3 block
scales [N][K / 16] (row major), and inverse [N] FP32, one over each row's global scale (a value is e2m1 * scale *
inverse). FP8: codes [N][K] E4M3 and FP32 row scales [N], or E8M0 bytes [N][K / 32] (MXFP8, modelopt's layout)."""
import torch

from ._ops import ops
from .triton_gemv import MAX_ROWS as TRITON_ROWS, decode, decode_fp8, gemv_triton, hardware
from .dot_gemv import MAX_ROWS as DOT_ROWS, gemv_fp8_dot, partial_elements as dot_partial_elements, \
    tile as fp8_dot_tile

CUDA_ROWS = 16   # rows the CUDA GEMVs take in one call

__all__ = ["CUDA_ROWS", "DOT_ROWS", "TRITON_ROWS", "dot_partial_elements", "fp8_dot_tile", "gemv", "gemv_fp8",
           "gemv_fp8_dot", "gemv_bf16", "gemv_triton", "decode", "decode_fp8", "hardware_decode", "hardware"]


def gemv(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, inverse: torch.Tensor, out: torch.Tensor,
         blocked: bool = False):
    """out [M][N] (FP32 or x's dtype) = x [M][K] (M 1..16, BF16 or FP16, rows on 16-byte boundaries, K a multiple of
    32) times the NVFP4 matrix, on CUDA cores. blocked: the E4M3 block scales are in cuBLASLt's 128 x 4 tiled layout
    (what its block-scaled GEMM reads; rows padded to 128, blocks to 4) instead of row major [N][K / 16]."""
    ops.gemv(x, codes, scales, inverse, blocked, out)
    return out


def gemv_fp8(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, out: torch.Tensor):
    """out [M][N] (FP32 or x's dtype) = x [M][K] (M 1..16, BF16 or FP16, rows on 16-byte boundaries) times the FP8
    matrix: codes [N][K] E4M3 (float8_e4m3fn or uint8, K a multiple of 16) with scales FP32 [N] (one per row, applied
    after the sum) or uint8 E8M0 [N][K / 32] (MXFP8, each value times its power of two), on CUDA cores."""
    ops.gemv_fp8(x, codes.view(torch.uint8), scales, out)
    return out


def gemv_bf16(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor):
    """out [M][N] (FP32 or BF16) = x [M][K] (M 1..16, BF16, rows on 16-byte boundaries) times w [N][K] BF16 (K a
    multiple of 8), on CUDA cores."""
    ops.gemv_bf16(x, w, out)
    return out


def hardware_decode(device=None) -> bool:
    """Whether the CUDA GEMV's code for `device` decodes E2M1 by the hardware conversion (architecture-specific
    targets from sm_100a on) rather than by the bit pattern; both give the same bits."""
    device = torch.device(device or "cuda")
    return bool(ops.hardware_decode(torch.empty(0, device=device)))
