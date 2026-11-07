"""Decode GEMVs from crit's quantized-projections technique, weight only: NVFP4 weights (W4A16), FP8 E4M3 weights with a
row scale or MXFP8's E8M0 scale per 32 inputs (W8A16) and BF16 weights times 1 to 16 rows of 16-bit activations on
CUDA cores; an NVFP4 GEMV for 1 to 8 rows in Triton; and the decoders that write an NVFP4 matrix's values for a
provider's GEMM. FP32 sums; each row's result is the same bits alone and beside the other rows of a call (the order
never depends on the row count). The CUDA and the Triton GEMV sum in different orders, so a process keeps to one.

NVFP4 matrices as checkpoints store them: codes [N][K / 2] uint8 (element 2i in the low nibble of byte i), E4M3 block
scales [N][K / 16] (row major), and inverse [N] FP32, one over each row's global scale (a value is e2m1 * scale *
inverse). FP8: codes [N][K] E4M3 and FP32 row scales [N], or E8M0 bytes [N][K / 32] (MXFP8, modelopt's layout)."""
import torch

from ._ops import ops
from .triton_gemv import MAX_ROWS as TRITON_ROWS, decode, decode_fp8, gemv_triton, hardware

CUDA_ROWS = 16   # rows the CUDA GEMVs take in one call
MAX_ROWS = TRITON_ROWS

__all__ = ["CUDA_ROWS", "TRITON_ROWS", "MAX_ROWS", "gemv", "gemv_fp8", "gemv_bf16", "gemv_triton", "decode",
           "decode_fp8", "hardware_decode", "hardware"]


def gemv(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, inverse: torch.Tensor, out: torch.Tensor):
    """out [M][N] (FP32 or x's dtype) = x [M][K] (M 1..16, BF16 or FP16, rows on 16-byte boundaries, K a multiple of
    32) times the NVFP4 matrix, on CUDA cores."""
    ops.gemv(x, codes, scales, inverse, out)
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
