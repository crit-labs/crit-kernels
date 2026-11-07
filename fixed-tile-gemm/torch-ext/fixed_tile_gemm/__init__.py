"""A row-invariant prompt GEMM from crit's quantized-projections technique: x [M][K] times dense [N][K] (both BF16 or
both FP16) on tensor cores with FP32 accumulation, in fixed tiles with one K loop whatever M is (no autotuning, no
split of K), so a row's result is the same bits whatever rows share the GEMM. torch's GEMM picks its kernel by the row
count and so does not promise it; with this one, a model can put the prompts of several sequences into one GEMM and
still give each the result of a GEMM of its own."""
import torch
import triton
import triton.language as tl

__all__ = ["mm", "MM_TILE", "MM_SMALL"]


@triton.jit
def _mm_kernel(x_ptr, w_ptr, out_ptr, M, N, K, stride_xm, stride_wn, stride_om,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    tiles_m, tiles_n = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * tiles_n)
    first = group * GROUP_M
    size = tl.minimum(tiles_m - first, GROUP_M)
    tm = first + (pid % (GROUP_M * tiles_n)) % size
    tn = (pid % (GROUP_M * tiles_n)) // size
    rm = tm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kmask = k0 + rk < K
        a = tl.load(x_ptr + rm[:, None] * stride_xm + (k0 + rk)[None, :], mask=(rm < M)[:, None] & kmask[None, :],
                    other=0.0)
        b = tl.load(w_ptr + rn[:, None] * stride_wn + (k0 + rk)[None, :], mask=(rn < N)[:, None] & kmask[None, :],
                    other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=(rm < M)[:, None] & (rn < N)[None, :])


# Two tiles of one K loop (BLOCK_K 64, the same dot): the small one where the large one would not fill the device
# twice over. The tile picks which outputs a program computes, never the order an output's sum takes, so a row's
# result is the same in either (tests/test_fixed_tile_gemm.py checks it across the switch).
MM_TILE = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 8, "num_stages": 3}
MM_SMALL = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 4, "num_stages": 4}

_SMS = {}


def _sms(device):
    """The device's multiprocessor count."""
    if device not in _SMS:
        _SMS[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[device]


def mm(x: torch.Tensor, dense: torch.Tensor, dtype=None) -> torch.Tensor:
    """x [M][K] times dense [N][K] (both BF16 or both FP16, unit column stride) -> out [M][N] in dtype (x's by
    default), FP32 accumulation, in MM_TILE's or MM_SMALL's tiles whatever M is."""
    M, K = x.shape
    N = dense.shape[0]
    if dense.shape[1] != K or dense.dtype != x.dtype or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"activations [{M}][{K}] {x.dtype} for a matrix [{N}][{dense.shape[1]}] {dense.dtype}: "
                         "both BF16 or both FP16")
    if x.stride(-1) != 1 or dense.stride(-1) != 1:
        raise ValueError("x and dense need unit column strides")
    out = torch.empty(M, N, dtype=dtype or x.dtype, device=x.device)
    t = MM_TILE if triton.cdiv(M, 128) * triton.cdiv(N, 128) >= 2 * _sms(x.device) else MM_SMALL
    grid = (triton.cdiv(M, t["BLOCK_M"]) * triton.cdiv(N, t["BLOCK_N"]),)
    _mm_kernel[grid](x, dense, out, M, N, K, x.stride(0), dense.stride(0), out.stride(0), BLOCK_M=t["BLOCK_M"],
                     BLOCK_N=t["BLOCK_N"], BLOCK_K=t["BLOCK_K"], GROUP_M=t["GROUP_M"], num_warps=t["num_warps"],
                     num_stages=t["num_stages"])
    return out
