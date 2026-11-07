"""The Triton side of decode-gemv: an NVFP4 W4A16 GEMV (1 to 8 rows, its own order) and the decoders that write a
matrix's values for a provider's GEMM (BF16/FP16, or FP8 rows in units of a row scale).

An E2M1 code's FP16 bit pattern (sign at bit 15, magnitude bits at 11..9) is its value times 2^-14; the kernels build
values that way and fold the 2^14 into the row factor. On a device with the hardware conversion (compute capability 10
and later: cvt.rn.f16x2.e2m1x2) the GEMV decodes with it instead, four bytes an instruction; both give the exact values,
so the results are the same bits either way.
"""
import torch
import triton
import triton.language as tl

MAX_ROWS = 8


@triton.jit
def _e2m1_pair(packed):
    """(low nibbles, high nibbles) of the bytes as FP16 values, exactly (no 2^-14 factor): the hardware conversion,
    four bytes an instruction group."""
    return tl.inline_asm_elementwise(
        """{
        .reg .b32 r0, r1, r2, r3;
        .reg .b8 b0, b1, b2, b3;
        mov.b32 {b0, b1, b2, b3}, $4;
        cvt.rn.f16x2.e2m1x2 r0, b0;
        cvt.rn.f16x2.e2m1x2 r1, b1;
        cvt.rn.f16x2.e2m1x2 r2, b2;
        cvt.rn.f16x2.e2m1x2 r3, b3;
        prmt.b32 $0, r0, r1, 0x5410;
        prmt.b32 $1, r2, r3, 0x5410;
        prmt.b32 $2, r0, r1, 0x7632;
        prmt.b32 $3, r2, r3, 0x7632;
        }""",
        "=r,=r,=r,=r,r", [packed], dtype=(tl.float16, tl.float16), is_pure=True, pack=4)


@triton.jit
def _e2m1(codes):
    bits = ((codes & 7).to(tl.uint16) << 9) | ((codes & 8).to(tl.uint16) << 12)
    return bits.to(tl.float16, bitcast=True).to(tl.float32)


# Tiles (rows per program, columns per step, warps) the GEMV picks from by measuring them on the device, once per
# matrix shape (N, K): never per row count, so a row reduces in the same order alone and in a batch.
GEMV_TILES = [triton.Config({"BLOCK_N": n, "BLOCK_K": k}, num_warps=w)
              for n, k, w in ((8, 512, 4), (4, 1024, 4), (4, 512, 2), (16, 512, 8), (8, 256, 4), (16, 256, 4),
                              (4, 256, 2), (8, 1024, 8), (2, 1024, 2), (32, 256, 8))]


@triton.jit
def _part(lo, hi, x_ptr, k0, j2, kmask, GROUPS: tl.constexpr):
    xx = tl.load(x_ptr + k0 + j2, mask=kmask, other=0.0).to(tl.float32)
    xe, xo = tl.split(tl.reshape(xx, (GROUPS, 8, 2)))
    return tl.sum(lo * xe[None, :, :] + hi * xo[None, :, :], axis=2)


@triton.autotune(configs=GEMV_TILES, key=["N", "K", "HW"])
@triton.jit
def _gemv_kernel(x_ptr, w_ptr, s_ptr, f_ptr, out_ptr, N, K, stride_xm, stride_om,
                 M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, HW: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    HALF: tl.constexpr = BLOCK_K // 2
    GROUPS: tl.constexpr = BLOCK_K // 16
    j = tl.arange(0, HALF)
    j2 = tl.arange(0, BLOCK_K)
    g = tl.arange(0, GROUPS)
    # One FP32 sum per row and scale group of the step, carried over the K loop: the groups' order across lanes is
    # reduced once, after it.
    acc0 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    acc1 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    acc2 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    acc3 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    acc4 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    acc5 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    acc6 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    acc7 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kmask = k0 + 2 * j < K
        kmask2 = k0 + j2 < K
        gmask = k0 + 16 * g < K
        packed = tl.load(w_ptr + rows[:, None] * (K // 2) + (k0 // 2 + j)[None, :],
                         mask=rmask[:, None] & kmask[None, :], other=0)
        scale = tl.load(s_ptr + rows[:, None] * (K // 16) + (k0 // 16 + g)[None, :],
                        mask=rmask[:, None] & gmask[None, :], other=0.0).to(tl.float32)
        if HW:
            lo16, hi16 = _e2m1_pair(packed)
            lo = tl.reshape(lo16.to(tl.float32), (BLOCK_N, GROUPS, 8))
            hi = tl.reshape(hi16.to(tl.float32), (BLOCK_N, GROUPS, 8))
        else:
            lo = tl.reshape(_e2m1(packed), (BLOCK_N, GROUPS, 8))
            hi = tl.reshape(_e2m1(packed >> 4), (BLOCK_N, GROUPS, 8))
        acc0 += _part(lo, hi, x_ptr, k0, j2, kmask2, GROUPS) * scale
        if M > 1:
            acc1 += _part(lo, hi, x_ptr + stride_xm, k0, j2, kmask2, GROUPS) * scale
        if M > 2:
            acc2 += _part(lo, hi, x_ptr + 2 * stride_xm, k0, j2, kmask2, GROUPS) * scale
        if M > 3:
            acc3 += _part(lo, hi, x_ptr + 3 * stride_xm, k0, j2, kmask2, GROUPS) * scale
        if M > 4:
            acc4 += _part(lo, hi, x_ptr + 4 * stride_xm, k0, j2, kmask2, GROUPS) * scale
        if M > 5:
            acc5 += _part(lo, hi, x_ptr + 5 * stride_xm, k0, j2, kmask2, GROUPS) * scale
        if M > 6:
            acc6 += _part(lo, hi, x_ptr + 6 * stride_xm, k0, j2, kmask2, GROUPS) * scale
        if M > 7:
            acc7 += _part(lo, hi, x_ptr + 7 * stride_xm, k0, j2, kmask2, GROUPS) * scale
    f = tl.load(f_ptr + rows, mask=rmask, other=0.0) * (1.0 if HW else 16384.0)
    tl.store(out_ptr + rows, (tl.sum(acc0, axis=1) * f).to(out_ptr.dtype.element_ty), mask=rmask)
    if M > 1:
        tl.store(out_ptr + stride_om + rows, (tl.sum(acc1, axis=1) * f).to(out_ptr.dtype.element_ty), mask=rmask)
    if M > 2:
        tl.store(out_ptr + 2 * stride_om + rows, (tl.sum(acc2, axis=1) * f).to(out_ptr.dtype.element_ty), mask=rmask)
    if M > 3:
        tl.store(out_ptr + 3 * stride_om + rows, (tl.sum(acc3, axis=1) * f).to(out_ptr.dtype.element_ty), mask=rmask)
    if M > 4:
        tl.store(out_ptr + 4 * stride_om + rows, (tl.sum(acc4, axis=1) * f).to(out_ptr.dtype.element_ty), mask=rmask)
    if M > 5:
        tl.store(out_ptr + 5 * stride_om + rows, (tl.sum(acc5, axis=1) * f).to(out_ptr.dtype.element_ty), mask=rmask)
    if M > 6:
        tl.store(out_ptr + 6 * stride_om + rows, (tl.sum(acc6, axis=1) * f).to(out_ptr.dtype.element_ty), mask=rmask)
    if M > 7:
        tl.store(out_ptr + 7 * stride_om + rows, (tl.sum(acc7, axis=1) * f).to(out_ptr.dtype.element_ty), mask=rmask)


_HW = {}


def hardware(device):
    """Whether the Triton GEMV decodes E2M1 by the hardware conversion on `device` (compute capability 10 and later;
    Triton builds for its architecture-specific target there). Tests set _HW[device] to compare both decodes."""
    device = torch.device(device)
    if device not in _HW:
        _HW[device] = torch.cuda.get_device_capability(device)[0] >= 10
    return _HW[device]


def gemv_triton(x, codes, scales, inverse, out):
    """x [M][K] (M <= 8, unit column stride) times the NVFP4 matrix (codes [N][K / 2] uint8, scales [N][K / 16] E4M3,
    inverse [N] FP32) -> out [M][N] (any float dtype, unit column stride), FP32 accumulation."""
    M, K = x.shape
    N = codes.shape[0]
    if not 1 <= M <= MAX_ROWS or K % 16 or codes.shape[1] * 2 != K or x.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError(f"gemv_triton: x [1..{MAX_ROWS}][K] (K a multiple of 16, unit column stride) for codes "
                         f"[N][K / 2], not x {tuple(x.shape)} and codes {tuple(codes.shape)}")
    _gemv_kernel[lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)](x, codes, scales, inverse, out, N, K, x.stride(0),
                                                                  out.stride(0), M=M, HW=hardware(x.device))
    return out


@triton.jit
def _decode_kernel(w_ptr, s_ptr, f_ptr, out_ptr, K, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    k0 = tl.program_id(1) * BLOCK_K
    HALF: tl.constexpr = BLOCK_K // 2
    j = tl.arange(0, HALF)
    kmask = k0 + 2 * j < K
    packed = tl.load(w_ptr + row * (K // 2) + k0 // 2 + j, mask=kmask, other=0)
    scale = tl.load(s_ptr + row * (K // 16) + (k0 + 2 * j) // 16, mask=kmask, other=0.0).to(tl.float32)
    f = tl.load(f_ptr + row) * 16384.0
    lo = _e2m1(packed) * scale * f
    hi = _e2m1(packed >> 4) * scale * f
    offs = k0 + tl.arange(0, BLOCK_K)
    tl.store(out_ptr + row * K + offs, tl.interleave(lo, hi).to(out_ptr.dtype.element_ty), mask=offs < K)


def decode(codes, scales, inverse, out):
    """The matrix's values [N][K] in out's dtype (16-bit, contiguous): e2m1 * scale * inverse, one FP32 product
    rounded once."""
    N, K = out.shape
    _decode_kernel[(N, triton.cdiv(K, 512))](codes, scales, inverse, out, K, BLOCK_K=512, num_warps=4)
    return out


@triton.jit
def _decode_fp8_kernel(w_ptr, s_ptr, f_ptr, out_ptr, K, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    k0 = tl.program_id(1) * BLOCK_K
    HALF: tl.constexpr = BLOCK_K // 2
    j = tl.arange(0, HALF)
    kmask = k0 + 2 * j < K
    packed = tl.load(w_ptr + row * (K // 2) + k0 // 2 + j, mask=kmask, other=0)
    scale = tl.load(s_ptr + row * (K // 16) + (k0 + 2 * j) // 16, mask=kmask, other=0.0).to(tl.float32)
    f = tl.load(f_ptr + row) * 16384.0
    lo = tl.clamp(_e2m1(packed) * scale * f, -448.0, 448.0)
    hi = tl.clamp(_e2m1(packed >> 4) * scale * f, -448.0, 448.0)
    offs = k0 + tl.arange(0, BLOCK_K)
    tl.store(out_ptr + row * K + offs, tl.interleave(lo, hi).to(tl.float8e4nv), mask=offs < K)


def decode_fp8(codes, scales, factor, out):
    """The matrix's values [N][K] as FP8 E4M3 (out contiguous float8_e4m3fn) in units of each row's scale: e2m1 *
    scale * factor[row], rounded once to nearest even within +-448."""
    N, K = out.shape
    _decode_fp8_kernel[(N, triton.cdiv(K, 512))](codes, scales, factor, out, K, BLOCK_K=512, num_warps=4)
    return out
