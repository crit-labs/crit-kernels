// Decode GEMVs on CUDA cores from crit's quantized-projections technique: weights stored as NVFP4, FP8 (E4M3 with a
// row scale, or MXFP8's E8M0 scale per 32 inputs) or BF16 times 1 to 16 rows of 16-bit activations (weight only:
// W4A16, W8A16), FP32 sums, each row's result the same bits alone and beside up to 15 others.
//
// NVFP4: out[m][n] = inverse[n] * sum_k x[m][k] * e2m1(codes[n][k]) * e4m3(scales[n][k / 16])
//   codes [N][K / 2] bytes (element 2i in the low nibble of byte i), scales [N][K / 16] E4M3 bytes, inverse [N] FP32
//   (one over the row's global scale), x [M][K] BF16 or FP16 with rows ldx elements apart (16-byte aligned, K a multiple
//   of 32), out [M][N] FP32 or x's type with rows ldo apart.
// FP8: out[m][n] = row_scale[n] * sum_k x[m][k] * e4m3(codes[n][k]), or (MXFP8) sum_k x[m][k] * e4m3(codes[n][k]) *
//   2^(mx[n][k / 32] - 127); codes [N][K] E4M3 bytes (K a multiple of 16, 32 for MXFP8), row_scale [N] FP32 or mx
//   [N][K / 32] E8M0 bytes (exactly one of the two), x and out as for NVFP4.
// BF16: out[m][n] = sum_k x[m][k] * w[n][k], w [N][K] BF16 (K a multiple of 8), x [M][K] BF16, out FP32 or BF16.
//
// Each warp takes a fixed number of output rows; each lane reads 16 bytes of every one of them per step. A weight's
// value (an E2M1 code times its E4M3 block scale, an E4M3 code, times a power-of-two block scale for MXFP8) is exact in
// FP32 and so is its product with an activation; each lane sums its products in FP32 in a fixed order and one
// xor-shuffle tree adds the lanes. Nothing of this depends on M or on the rows a warp takes, so a row's result does not
// depend on the rows beside it. E2M1 codes decode by the hardware conversion (cvt.rn.f16x2.e2m1x2) on the
// architecture-specific targets that have it (sm_100a and later) and by their FP16 bit pattern elsewhere: exact either
// way, so both give the same bits.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace crit::kernels::decode_gemv {

constexpr int kMaxRows = 16;

cudaError_t enqueue_gemv(const void* x, int ldx, bool x_bf16, const std::uint8_t* codes, const std::uint8_t* scales,
                         const float* inverse, void* out, int ldo, bool out_fp32, int M, int N, int K,
                         cudaStream_t stream);

cudaError_t enqueue_gemv_fp8(const void* x, int ldx, bool x_bf16, const std::uint8_t* codes, const float* row_scale,
                             const std::uint8_t* mx_scale, void* out, int ldo, bool out_fp32, int M, int N, int K,
                             cudaStream_t stream);

cudaError_t enqueue_gemv_bf16(const void* x, int ldx, const void* w, void* out, int ldo, bool out_fp32, int M, int N,
                              int K, cudaStream_t stream);

// Whether this build decodes E2M1 by the hardware conversion on the current device's target (1) or by the bit
// pattern (0): the two give the same bits; tests compare them.
int hardware_decode();

}  // namespace crit::kernels::decode_gemv
