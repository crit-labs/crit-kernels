"""A Triton FP8 W8A16 decode GEMV on tensor cores for 1 to 16 rows: a program multiplies a [16 rows x BK] block of the
activations by a [BN x BK] block of the E4M3 weights (exact in BF16) with tl.dot, FP32 sums, over its part of K; with
`split` parts the programs write FP32 partial sums and a second kernel adds them in part order and applies the row
scale. The program's shape never depends on the call's row count (rows past it are masked to zero), so a row's bits
are the same alone and beside up to 15 others; tile() picks the shape from the matrix's [N, K] and the device's
multiprocessor count, so a process keeps one order a matrix. Adapted from a crit-labs engine's Triton decode GEMV
(Apache-2.0), as general code."""
import torch
import triton
import triton.language as tl

MAX_ROWS = 16


@triton.jit
def _fp8_dot_kernel(x_ptr, w_ptr, s_ptr, out_ptr, T, N, K, KS, x_stride, out_stride,
                    BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    rows = tl.arange(0, 16)
    n = pid_n * BN + tl.arange(0, BN)
    nmask = n < N
    nn = tl.minimum(n, N - 1)   # a row past N reads row N - 1 and writes nothing
    acc = tl.zeros([16, BN], dtype=tl.float32)
    xmask = rows[:, None] < T
    for k0 in range(pid_s * KS, pid_s * KS + KS, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rows[:, None] * x_stride + k[None, :], mask=xmask, other=0.)
        w = tl.load(w_ptr + nn[:, None].to(tl.int64) * K + k[None, :])
        acc = tl.dot(x.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)), acc)
    omask = xmask & nmask[None, :]
    if SPLIT == 1:
        y = acc * tl.load(s_ptr + nn)[None, :]
        tl.store(out_ptr + rows[:, None] * out_stride + n[None, :], y.to(out_ptr.dtype.element_ty), mask=omask)
    else:
        tl.store(out_ptr + pid_s * T * N + rows[:, None] * N + n[None, :], acc, mask=omask)


@triton.jit
def _reduce_kernel(part_ptr, s_ptr, out_ptr, T, N, out_stride, SPLIT: tl.constexpr, BN: tl.constexpr):
    t = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    nmask = n < N
    acc = tl.load(part_ptr + t * N + n, mask=nmask, other=0.)
    for s in tl.static_range(1, SPLIT):
        acc += tl.load(part_ptr + s * T * N + t * N + n, mask=nmask, other=0.)
    y = acc * tl.load(s_ptr + n, mask=nmask, other=0.)
    tl.store(out_ptr + t * out_stride + n, y.to(out_ptr.dtype.element_ty), mask=nmask)


def tile(n: int, k: int, multiprocessors: int):
    """(BN, BK, warps, split) for an [N, K] FP8 matrix, from its shape and the multiprocessor count alone: 64 rows and
    8 warps a program from 16384 rows on, else 32 and 4; BK 256 (512 from 65536 rows on, 128 where 256 does not divide
    K); K split in two, four, ... while the programs are fewer than the multiprocessors and each part keeps at least 4
    BK. (Measured on an RTX 5090 over ten FP8 matrices of three models, 640 to 248320 rows, at 1 to 16 rows: within 1-6%
    of the best tile of a sweep.)"""
    if k % 128:
        raise ValueError(f"the FP8 dot GEMV takes K a multiple of 128, not {k}")
    bn, warps = (64, 8) if n >= 16384 else (32, 4)
    bk = 512 if n >= 65536 and k % 512 == 0 else 256 if k % 256 == 0 else 128
    split = 1
    while triton.cdiv(n, bn) * split < multiprocessors and k % (2 * split * bk) == 0 and k // (2 * split) >= 4 * bk:
        split *= 2
    return bn, bk, warps, split


def partial_elements(rows: int, n: int, split: int) -> int:
    """FP32 elements the partial sums of a call take (0 for one part)."""
    return split * rows * n if split > 1 else 0


def gemv_fp8_dot(x: torch.Tensor, w: torch.Tensor, row_scale: torch.Tensor, out: torch.Tensor, plan, partial=None):
    """out [M][N] (any float dtype, unit column stride) = x [M][K] (M 1..16, BF16 or FP16, unit column stride) times
    w [N][K] E4M3 (float8_e4m3fn or uint8) with row_scale [N] FP32 after the sum; plan: tile()'s; partial: FP32 with
    partial_elements(M, N, split) elements or more when split > 1 (a call allocates nothing then)."""
    bn, bk, warps, split = plan
    M, K = x.shape
    N = w.shape[0]
    if not 1 <= M <= MAX_ROWS or w.shape[1] != K or K % (split * bk) or x.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError(f"gemv_fp8_dot: x [1..{MAX_ROWS}][K] for w [N][K] with K a multiple of split * BK "
                         f"({split} x {bk}), not x {tuple(x.shape)} and w {tuple(w.shape)}")
    w = w.view(torch.float8_e4m3fn)
    grid = (triton.cdiv(N, bn), split)
    if split == 1:
        _fp8_dot_kernel[grid](x, w, row_scale, out, M, N, K, K, x.stride(0), out.stride(0), BN=bn, BK=bk, SPLIT=1,
                              num_warps=warps)
        return out
    need = partial_elements(M, N, split)
    if partial is None or partial.numel() < need or partial.dtype != torch.float32:
        raise ValueError(f"gemv_fp8_dot: {split} parts of K take an FP32 partial buffer of {need} elements or more")
    _fp8_dot_kernel[grid](x, w, row_scale, partial, M, N, K, K // split, x.stride(0), out.stride(0), BN=bn, BK=bk,
                          SPLIT=split, num_warps=warps)
    _reduce_kernel[(triton.cdiv(N, 1024), M)](partial, row_scale, out, M, N, out.stride(0), SPLIT=split, BN=1024,
                                              num_warps=4)
    return out
