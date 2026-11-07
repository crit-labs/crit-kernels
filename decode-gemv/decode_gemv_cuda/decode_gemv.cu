// Decode GEMVs on CUDA cores (decode_gemv.h): NVFP4 (W4A16) and BF16 weights times 1 to 8 rows of 16-bit activations.
#include "decode_gemv.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// The conversion exists on the architecture-specific and family targets from sm_100 on (100a/f, 103a, 110a/f,
// 120a/f, 121a), not on plain sm_100 or sm_120 nor before.
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000
#define CRIT_E2M1_HW 1
#else
#define CRIT_E2M1_HW 0
#endif

namespace crit::kernels::decode_gemv {
namespace {

constexpr int WARPS = 4;

// Eight codes of a 32-bit word as values (in element order), times `scale` (the block scale, times 2^14 on the bit
// pattern path: an E2M1 code's FP16 bit pattern, sign at bit 15 and magnitude bits at 11..9, is its value times 2^-14).
__device__ __forceinline__ void decode8(uint32_t word, float scale, float* w) {
  uint32_t h[4];
#if CRIT_E2M1_HW
  asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
      "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
      "cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
      "cvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
      : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]) : "r"(word));
#else
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const uint32_t byte = (word >> (8 * b)) & 0xffu;
    const uint32_t lo = ((byte & 7u) << 9) | ((byte & 8u) << 12);
    const uint32_t hi = (((byte >> 4) & 7u) << 9) | (((byte >> 4) & 8u) << 12);
    h[b] = lo | (hi << 16);
  }
#endif
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h[b]));
    w[2 * b] = f.x * scale;
    w[2 * b + 1] = f.y * scale;
  }
}

__device__ __forceinline__ float block_scale(uint8_t bits) {
  __nv_fp8_e4m3 v;
  v.__x = bits;
  return CRIT_E2M1_HW ? float(v) : float(v) * 16384.0f;
}

__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

template <typename X>
__device__ __forceinline__ void load8(const X* p, float* out) {
  const uint4 a = __ldg(reinterpret_cast<const uint4*>(p));
  const X* va = reinterpret_cast<const X*>(&a);
#pragma unroll
  for (int e = 0; e < 8; ++e) out[e] = to_float(va[e]);
}

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half_rn(v); }

template <int M, int ROWS, typename X, typename O>
__global__ void __launch_bounds__(WARPS * 32) gemv_kernel(const X* __restrict__ x, int ldx,
                                                          const uint8_t* __restrict__ codes,
                                                          const uint8_t* __restrict__ scales,
                                                          const float* __restrict__ inverse, O* __restrict__ out,
                                                          int ldo, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * ROWS;
  if (row0 >= N) return;
  const int kb = K >> 1, ks = K >> 4;
  float acc[ROWS][M];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
  for (int off = lane * 16; off < kb; off += 512) {
    uint4 wv[ROWS];
    uint16_t sv[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int n = min(row0 + r, N - 1);   // a row past N reads row N - 1 and writes nothing
      wv[r] = __ldg(reinterpret_cast<const uint4*>(codes + (size_t)n * kb + off));
      sv[r] = __ldg(reinterpret_cast<const uint16_t*>(scales + (size_t)n * ks + (off >> 3)));
    }
    const int k = off * 2;
    // The lane's four words in element order, 8 codes each (two to a scale block): each row's sums take the same
    // order whatever ROWS and M are.
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float w[ROWS][8];
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const uint32_t* words = reinterpret_cast<const uint32_t*>(&wv[r]);
        decode8(words[q], block_scale(uint8_t(q >= 2 ? (sv[r] >> 8) : (sv[r] & 0xff))), w[r]);
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float xf[8];
        load8(x + (size_t)m * ldx + k + 8 * q, xf);
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[r][m] = fmaf(w[r][e], xf[e], acc[r][m]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int n = row0 + r;
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0 && n < N) out[(size_t)m * ldo + n] = from_float<O>(v * inverse[n]);
    }
  }
}

// The current device's multiprocessor count, per device (queried once each).
int multiprocessors() {
  static int counts[64] = {};
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) return 1;
  if (!counts[device]) cudaDeviceGetAttribute(&counts[device], cudaDevAttrMultiProcessorCount, device);
  return counts[device] > 0 ? counts[device] : 1;
}

template <int M, int ROWS, typename X, typename O>
void launch_rows(const void* x, int ldx, const uint8_t* codes, const uint8_t* scales, const float* inverse, void* out,
                 int ldo, int N, int K, cudaStream_t stream) {
  const int blocks = (N + ROWS * WARPS - 1) / (ROWS * WARPS);
  gemv_kernel<M, ROWS, X, O><<<blocks, WARPS * 32, 0, stream>>>(reinterpret_cast<const X*>(x), ldx, codes, scales,
                                                               inverse, reinterpret_cast<O*>(out), ldo, N, K);
}

// Rows a warp takes, a choice of speed and never of a row's order: more at more activation rows (each activation
// converted once for them all), fewer where the matrix would not give the device two blocks a multiprocessor.
template <int M, typename X, typename O>
void launch(const void* x, int ldx, const uint8_t* codes, const uint8_t* scales, const float* inverse, void* out,
            int ldo, int N, int K, cudaStream_t stream) {
  constexpr int MORE = M <= 2 ? 2 : M <= 4 ? 4 : M <= 8 ? 8 : 4;
  constexpr int FEWER = MORE == 2 ? 2 : MORE / 2;
  if ((N + MORE * WARPS - 1) / (MORE * WARPS) >= 2 * multiprocessors())
    launch_rows<M, MORE, X, O>(x, ldx, codes, scales, inverse, out, ldo, N, K, stream);
  else
    launch_rows<M, FEWER, X, O>(x, ldx, codes, scales, inverse, out, ldo, N, K, stream);
}

template <typename X, typename O>
bool dispatch(int M, const void* x, int ldx, const uint8_t* codes, const uint8_t* scales, const float* inverse,
              void* out, int ldo, int N, int K, cudaStream_t stream) {
#define CALL(m) launch<m, X, O>(x, ldx, codes, scales, inverse, out, ldo, N, K, stream)
  switch (M) {
#define CRIT_ROWS_CASE(m) case m: CALL(m); return true;
    CRIT_ROWS_CASE(1) CRIT_ROWS_CASE(2) CRIT_ROWS_CASE(3) CRIT_ROWS_CASE(4) CRIT_ROWS_CASE(5) CRIT_ROWS_CASE(6)
    CRIT_ROWS_CASE(7) CRIT_ROWS_CASE(8) CRIT_ROWS_CASE(9) CRIT_ROWS_CASE(10) CRIT_ROWS_CASE(11) CRIT_ROWS_CASE(12)
    CRIT_ROWS_CASE(13) CRIT_ROWS_CASE(14) CRIT_ROWS_CASE(15) CRIT_ROWS_CASE(16)
#undef CRIT_ROWS_CASE
    default: return false;
  }
#undef CALL
}

// BF16: each lane reads 16 bytes (8 weights) of every row per step; each product is exact in FP32 (8 by 8 significant
// bits), a lane sums in a fixed order and one xor-shuffle tree adds the lanes.
template <int M, int ROWS, typename O>
__global__ void __launch_bounds__(WARPS * 32) bf16_kernel(const __nv_bfloat16* __restrict__ x, int ldx,
                                                          const __nv_bfloat16* __restrict__ w, O* __restrict__ out,
                                                          int ldo, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * ROWS;
  if (row0 >= N) return;
  float acc[ROWS][M];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
  for (int k = lane * 8; k < K; k += 256) {
    float wf[ROWS][8];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int n = min(row0 + r, N - 1);   // a row past N reads row N - 1 and writes nothing
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(w + (size_t)n * K + k));
      const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
      for (int e = 0; e < 8; ++e) wf[r][e] = to_float(b[e]);
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)m * ldx + k));
      const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
      float xf[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) xf[e] = to_float(b[e]);
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[r][m] = fmaf(wf[r][e], xf[e], acc[r][m]);
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int n = row0 + r;
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0 && n < N) out[(size_t)m * ldo + n] = from_float<O>(v);
    }
  }
}

template <int M, typename O>
void launch_bf16(const void* x, int ldx, const void* w, void* out, int ldo, int N, int K, cudaStream_t stream) {
  constexpr int ROWS = M <= 2 ? 2 : M <= 8 ? 4 : 2;   // rows a warp takes: speed only, never a row's order
  const int blocks = (N + ROWS * WARPS - 1) / (ROWS * WARPS);
  bf16_kernel<M, ROWS, O><<<blocks, WARPS * 32, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x), ldx,
                                                             reinterpret_cast<const __nv_bfloat16*>(w),
                                                             reinterpret_cast<O*>(out), ldo, N, K);
}

template <typename O>
bool dispatch_bf16(int M, const void* x, int ldx, const void* w, void* out, int ldo, int N, int K,
                   cudaStream_t stream) {
#define CALL(m) launch_bf16<m, O>(x, ldx, w, out, ldo, N, K, stream)
  switch (M) {
#define CRIT_ROWS_CASE(m) case m: CALL(m); return true;
    CRIT_ROWS_CASE(1) CRIT_ROWS_CASE(2) CRIT_ROWS_CASE(3) CRIT_ROWS_CASE(4) CRIT_ROWS_CASE(5) CRIT_ROWS_CASE(6)
    CRIT_ROWS_CASE(7) CRIT_ROWS_CASE(8) CRIT_ROWS_CASE(9) CRIT_ROWS_CASE(10) CRIT_ROWS_CASE(11) CRIT_ROWS_CASE(12)
    CRIT_ROWS_CASE(13) CRIT_ROWS_CASE(14) CRIT_ROWS_CASE(15) CRIT_ROWS_CASE(16)
#undef CRIT_ROWS_CASE
    default: return false;
  }
#undef CALL
}

template <typename X>
__device__ __forceinline__ void load16(const X* p, float* out) {
  const uint4 a = __ldg(reinterpret_cast<const uint4*>(p));
  const uint4 b = __ldg(reinterpret_cast<const uint4*>(p) + 1);
  const X* va = reinterpret_cast<const X*>(&a);
  const X* vb = reinterpret_cast<const X*>(&b);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    out[e] = to_float(va[e]);
    out[8 + e] = to_float(vb[e]);
  }
}

// An E8M0 byte as its FP32 scale 2^(e - 127) (0 is 2^-127; 255, NaN, never comes from a checkpoint).
__device__ __forceinline__ float e8m0(uint32_t e) {
  return __uint_as_float(e == 0u ? 0x00400000u : e == 255u ? 0x7fc00000u : e << 23);
}

// FP8 E4M3 weights (W8A16): out[m][n] = row[n] * sum_k x[m][k] * e4m3(codes[n][k]) (row scales), or sum_k x[m][k] *
// e4m3(codes[n][k]) * 2^(e[n][k / 32] - 127) (MXFP8). Each lane reads 16 codes of every row per step; an E4M3 value
// times an activation is exact in FP32 (4 by 8 significant bits) and so is a value times its power-of-two block scale;
// a lane sums in a fixed order, one xor-shuffle tree adds the lanes, and the row scale multiplies once at the end.
template <int M, int ROWS, typename X, typename O>
__global__ void __launch_bounds__(WARPS * 32) fp8_kernel(const X* __restrict__ x, int ldx,
                                                         const uint8_t* __restrict__ codes,
                                                         const float* __restrict__ row_scale,
                                                         const uint8_t* __restrict__ mx_scale, O* __restrict__ out,
                                                         int ldo, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * ROWS;
  if (row0 >= N) return;
  const int kb = K >> 5;
  float acc[ROWS][M];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
  for (int k = lane * 16; k < K; k += 512) {
    float w[ROWS][16];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int n = min(row0 + r, N - 1);   // a row past N reads row N - 1 and writes nothing
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(codes + (size_t)n * K + k));
      const __nv_fp8x2_storage_t* pairs = reinterpret_cast<const __nv_fp8x2_storage_t*>(&v);
      const float s = mx_scale ? e8m0(__ldg(mx_scale + (size_t)n * kb + (k >> 5))) : 1.0f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(pairs[e], __NV_E4M3);
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
        w[r][2 * e] = f.x * s;
        w[r][2 * e + 1] = f.y * s;
      }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float xf[16];
      load16(x + (size_t)m * ldx + k, xf);
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int e = 0; e < 16; ++e) acc[r][m] = fmaf(w[r][e], xf[e], acc[r][m]);
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int n = row0 + r;
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0 && n < N) out[(size_t)m * ldo + n] = from_float<O>(row_scale ? v * row_scale[n] : v);
    }
  }
}

template <int M, int ROWS, typename X, typename O>
void launch_fp8_rows(const void* x, int ldx, const uint8_t* codes, const float* row_scale, const uint8_t* mx_scale,
                     void* out, int ldo, int N, int K, cudaStream_t stream) {
  const int blocks = (N + ROWS * WARPS - 1) / (ROWS * WARPS);
  fp8_kernel<M, ROWS, X, O><<<blocks, WARPS * 32, 0, stream>>>(reinterpret_cast<const X*>(x), ldx, codes, row_scale,
                                                              mx_scale, reinterpret_cast<O*>(out), ldo, N, K);
}

// Rows a warp takes, speed only: fewer at more activation rows (each weight is decoded once for them all) and where
// the matrix would not give the device two blocks a multiprocessor.
template <int M, typename X, typename O>
void launch_fp8(const void* x, int ldx, const uint8_t* codes, const float* row_scale, const uint8_t* mx_scale,
                void* out, int ldo, int N, int K, cudaStream_t stream) {
  constexpr int MORE = M <= 4 ? 4 : M <= 8 ? 2 : 1;
  constexpr int FEWER = MORE == 1 ? 1 : MORE / 2;
  if ((N + MORE * WARPS - 1) / (MORE * WARPS) >= 2 * multiprocessors())
    launch_fp8_rows<M, MORE, X, O>(x, ldx, codes, row_scale, mx_scale, out, ldo, N, K, stream);
  else
    launch_fp8_rows<M, FEWER, X, O>(x, ldx, codes, row_scale, mx_scale, out, ldo, N, K, stream);
}

template <typename X, typename O>
bool dispatch_fp8(int M, const void* x, int ldx, const uint8_t* codes, const float* row_scale,
                  const uint8_t* mx_scale, void* out, int ldo, int N, int K, cudaStream_t stream) {
#define CALL(m) launch_fp8<m, X, O>(x, ldx, codes, row_scale, mx_scale, out, ldo, N, K, stream)
  switch (M) {
#define CRIT_ROWS_CASE(m) case m: CALL(m); return true;
    CRIT_ROWS_CASE(1) CRIT_ROWS_CASE(2) CRIT_ROWS_CASE(3) CRIT_ROWS_CASE(4) CRIT_ROWS_CASE(5) CRIT_ROWS_CASE(6)
    CRIT_ROWS_CASE(7) CRIT_ROWS_CASE(8) CRIT_ROWS_CASE(9) CRIT_ROWS_CASE(10) CRIT_ROWS_CASE(11) CRIT_ROWS_CASE(12)
    CRIT_ROWS_CASE(13) CRIT_ROWS_CASE(14) CRIT_ROWS_CASE(15) CRIT_ROWS_CASE(16)
#undef CRIT_ROWS_CASE
    default: return false;
  }
#undef CALL
}

__global__ void hardware_kernel(int* out) { *out = CRIT_E2M1_HW; }

}  // namespace

cudaError_t enqueue_gemv(const void* x, int ldx, bool x_bf16, const std::uint8_t* codes, const std::uint8_t* scales,
                         const float* inverse, void* out, int ldo, bool out_fp32, int M, int N, int K,
                         cudaStream_t stream) {
  if (M < 1 || M > kMaxRows || N < 1 || K < 32 || K % 32) return cudaErrorInvalidValue;
  bool ok;
  if (out_fp32) {
    ok = x_bf16 ? dispatch<__nv_bfloat16, float>(M, x, ldx, codes, scales, inverse, out, ldo, N, K, stream)
                : dispatch<__half, float>(M, x, ldx, codes, scales, inverse, out, ldo, N, K, stream);
  } else {
    ok = x_bf16 ? dispatch<__nv_bfloat16, __nv_bfloat16>(M, x, ldx, codes, scales, inverse, out, ldo, N, K, stream)
                : dispatch<__half, __half>(M, x, ldx, codes, scales, inverse, out, ldo, N, K, stream);
  }
  return ok ? cudaGetLastError() : cudaErrorInvalidValue;
}

cudaError_t enqueue_gemv_bf16(const void* x, int ldx, const void* w, void* out, int ldo, bool out_fp32, int M, int N,
                              int K, cudaStream_t stream) {
  if (M < 1 || M > kMaxRows || N < 1 || K < 8 || K % 8) return cudaErrorInvalidValue;
  const bool ok = out_fp32 ? dispatch_bf16<float>(M, x, ldx, w, out, ldo, N, K, stream)
                           : dispatch_bf16<__nv_bfloat16>(M, x, ldx, w, out, ldo, N, K, stream);
  return ok ? cudaGetLastError() : cudaErrorInvalidValue;
}

cudaError_t enqueue_gemv_fp8(const void* x, int ldx, bool x_bf16, const std::uint8_t* codes, const float* row_scale,
                             const std::uint8_t* mx_scale, void* out, int ldo, bool out_fp32, int M, int N, int K,
                             cudaStream_t stream) {
  if (M < 1 || M > kMaxRows || N < 1 || K < 16 || K % 16 || (mx_scale && K % 32) || (!row_scale == !mx_scale))
    return cudaErrorInvalidValue;
  bool ok;
  if (out_fp32) {
    ok = x_bf16 ? dispatch_fp8<__nv_bfloat16, float>(M, x, ldx, codes, row_scale, mx_scale, out, ldo, N, K, stream)
                : dispatch_fp8<__half, float>(M, x, ldx, codes, row_scale, mx_scale, out, ldo, N, K, stream);
  } else {
    ok = x_bf16 ? dispatch_fp8<__nv_bfloat16, __nv_bfloat16>(M, x, ldx, codes, row_scale, mx_scale, out, ldo, N, K,
                                                             stream)
                : dispatch_fp8<__half, __half>(M, x, ldx, codes, row_scale, mx_scale, out, ldo, N, K, stream);
  }
  return ok ? cudaGetLastError() : cudaErrorInvalidValue;
}

int hardware_decode() {
  int* found = nullptr;
  int value = -1;
  if (cudaMalloc(&found, sizeof(int)) != cudaSuccess) return -1;
  hardware_kernel<<<1, 1>>>(found);
  if (cudaMemcpy(&value, found, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) value = -1;
  cudaFree(found);
  return value;
}

}  // namespace crit::kernels::decode_gemv
