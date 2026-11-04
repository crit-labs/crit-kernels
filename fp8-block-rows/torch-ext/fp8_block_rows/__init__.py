"""Matched FP8 projections in crit's declared per-output order: E4M3 weights (MXFP8's E8M0 block scales, an FP8
checkpoint's 128 x 128 FP32 block scales, or row scales) against E4M3 activations quantized per group. Each 32-element
block is one e4m3 tensor-core MMA with a zero accumulator; the FP32 step (lane b % 16 = fma(block sum, weight scale *
activation scale, lane), the halving tree, post) is declared, so the decode rows and the prompt GEMM give the same bits
for every output: decode, verify rounds, prefill and a prompt cache agree bit for bit. Same bits on one GPU
architecture; check on the target (tests/fp8_block_tests.cu)."""
import struct
from typing import Sequence

import torch

from ._ops import ops

__all__ = ["matrix_table", "row_jobs", "gemm_jobs", "rows", "gemm", "quantize", "MX", "AMAX"]

MX, AMAX = 0, 1                          # activation scale kinds: MXFP8's 2^(floor(log2 amax) - 8), or amax / 448
_MATRIX = struct.Struct("<QQiiiifi")     # Matrix: codes, scales, rows, cols, scale_rows, scale_k, post, scale_e8m0
_ROW_JOB = struct.Struct("<ii8i8i")      # RowJob: matrix, tokens, act_row[8], out_row[8]
_GEMM_JOB = struct.Struct("<iii")        # GemmJob: matrix, first, count


def _device_bytes(packed: bytes, device) -> torch.Tensor:
    return torch.frombuffer(bytearray(packed), dtype=torch.uint8).to(device)


def matrix_table(matrices: Sequence, device=None) -> torch.Tensor:
    """One Matrix per (codes [rows, cols] uint8 E4M3, scales, scale_rows, scale_k, post) of CUDA tensors. scales are
    fp32 [ceil(rows / scale_rows), cols / scale_k], or uint8 E8M0 bytes as an MXFP8 checkpoint stores them. The table
    holds the tensors' pointers: keep them alive while it is used."""
    packed = b""
    for codes, scales, scale_rows, scale_k, post in matrices:
        if scales.dtype not in (torch.float32, torch.uint8):
            raise ValueError("scales are fp32 or uint8 E8M0")
        packed += _MATRIX.pack(codes.data_ptr(), scales.data_ptr(), codes.size(0), codes.size(1), int(scale_rows),
                               int(scale_k), float(post), int(scales.dtype == torch.uint8))
    return _device_bytes(packed, device or matrices[0][0].device)


def row_jobs(jobs: Sequence, device) -> torch.Tensor:
    """Decode jobs: (matrix, [activation rows], [output rows]) with at most 8 tokens each."""
    packed = b""
    for matrix, act_rows, out_rows in jobs:
        if not 0 < len(act_rows) <= 8 or len(act_rows) != len(out_rows):
            raise ValueError("a decode job has 1 to 8 tokens, one output row each")
        pad = [0] * (8 - len(act_rows))
        packed += _ROW_JOB.pack(matrix, len(act_rows), *act_rows, *pad, *out_rows, *pad)
    return _device_bytes(packed, device)


def gemm_jobs(jobs: Sequence, device) -> torch.Tensor:
    """Prompt jobs: (matrix, first activation row, count): out[first + i] = matrix applied to activation row first + i."""
    return _device_bytes(b"".join(_GEMM_JOB.pack(m, first, count) for m, first, count in jobs), device)


def rows(table, jobs, max_rows, codes, scales, group, out):
    ops.rows(table, jobs, max_rows, codes, scales, group, out)


def gemm(table, jobs, max_rows, max_count, codes, scales, group, out):
    ops.gemm(table, jobs, max_rows, max_count, codes, scales, group, out)


def quantize(x: torch.Tensor, group: int = 32, kind: int = MX):
    """x [rows, cols] fp32 or bf16 -> (codes uint8 E4M3, scales fp32 per group): crit's declared quantizer."""
    codes = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    scales = torch.empty(x.size(0), x.size(1) // group, dtype=torch.float32, device=x.device)
    ops.quantize(x.contiguous(), group, kind, codes, scales)
    return codes, scales
