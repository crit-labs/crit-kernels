"""Integer block expert projections in crit's declared integer order: 4-bit weights (NVFP4's E2M1 codes or INT4, E4M3
scale per 16) against activations quantized per block (E2M1 x 2 or INT8). Each 16-block's integer dot is exact; the
FP32 step (lane b % 16 = fma(sum, weight scale * activation scale, lane), the halving tree, post) is declared, so the
decode rows, the prompt GEMM and crit's CPU rows (host_int.h) give the same bits for every output: a CPU expert tier,
a device cache and the prompt path agree bit for bit. The quantizers and SwiGLU are crit's declared ones."""
import struct
from typing import Sequence

import torch

from ._ops import ops

__all__ = ["matrix_table", "row_jobs", "gemm_jobs", "rows", "gemm", "quantize_e2m1x2", "quantize_int8", "swiglu"]

_MATRIX = struct.Struct("<QQiifi")       # Matrix: codes, scales, rows, cols, post, int4
_ROW_JOB = struct.Struct("<ii8i8i")      # RowJob: matrix, tokens, act_row[8], out_row[8]
_GEMM_JOB = struct.Struct("<iii")        # GemmJob: matrix, first, count


def _device_bytes(packed: bytes, device) -> torch.Tensor:
    return torch.frombuffer(bytearray(packed), dtype=torch.uint8).to(device)


def matrix_table(matrices: Sequence, device=None) -> torch.Tensor:
    """One Matrix per (codes [rows, cols / 2] uint8, scales [rows, cols / 16] uint8 E4M3, post, int4) of CUDA tensors."""
    packed = b"".join(_MATRIX.pack(codes.data_ptr(), scales.data_ptr(), codes.size(0), codes.size(1) * 2, float(post),
                                   int(bool(int4))) for codes, scales, post, int4 in matrices)
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


def rows(table, jobs, max_rows, values, scales, block, out):
    ops.rows(table, jobs, max_rows, values, scales, block, out)


def gemm(table, jobs, max_rows, max_count, values, scales, block, out):
    ops.gemm(table, jobs, max_rows, max_count, values, scales, block, out)


def quantize_e2m1x2(x: torch.Tensor, global_scale: float):
    """x [rows, cols] fp32 -> (values int8 E2M1 x 2, scales fp32 per 16): crit's declared quantizer."""
    values = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty(x.size(0), x.size(1) // 16, dtype=torch.float32, device=x.device)
    ops.quantize_e2m1x2(x.contiguous(), float(global_scale), values, scales)
    return values, scales


def quantize_int8(x: torch.Tensor, block: int = 32):
    """x [rows, cols] fp32 -> (values int8, scales fp32 per block of 16 or 32): crit's declared quantizer."""
    values = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty(x.size(0), x.size(1) // block, dtype=torch.float32, device=x.device)
    ops.quantize_int8(x.contiguous(), block, values, scales)
    return values, scales


def swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """gate_up [rows, 2 * inter] fp32 -> [rows, inter] fp32, crit's declared SwiGLU."""
    out = torch.empty(gate_up.size(0), gate_up.size(1) // 2, dtype=torch.float32, device=gate_up.device)
    ops.swiglu(gate_up.contiguous(), out)
    return out
