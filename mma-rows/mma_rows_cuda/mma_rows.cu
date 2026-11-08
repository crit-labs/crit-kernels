// Projections of 1 to 64 rows on tensor cores, one fixed tile, and the activations' quantizers (mma_rows.h).
#include "mma_rows.h"
#include "fp4_mma.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// The E2M1 conversion exists on the architecture-specific and family targets from sm_100 on (100a/f, 103a, 110a/f,
// 120a/f, 121a), not on plain sm_100 or sm_120 nor before; elsewhere codes decode by their bit pattern (same values).
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000
#define CRIT_E2M1_HW 1
#else
#define CRIT_E2M1_HW 0
#endif

namespace crit::kernels::mma_rows {
namespace {

// fp32(1 / 448) and fp32(1 / 6): the quantizers multiply by them, as a division by a constant is done in torch.
constexpr float kRcp448 = 1.0f / 448.0f;
constexpr float kRcp6 = 1.0f / 6.0f;

__device__ __forceinline__ uint32_t bf16x2(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// The larger of two values, NaN if either is (as torch's amax).
__device__ __forceinline__ float nanmax(float a, float b) { return (a > b || a != a) ? a : b; }

__device__ __forceinline__ float e4m3(uint32_t bits) {
  __nv_fp8_e4m3 v;
  v.__x = static_cast<__nv_fp8_storage_t>(bits & 0xffu);
  return float(v);
}
__device__ __forceinline__ uint32_t e4m3_bits(float v) {   // round to nearest even, saturating (v is in range)
  return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float e8m0(uint32_t e) {
  return e == 255u ? __uint_as_float(0x7fc00000u) : e ? __uint_as_float(e << 23) : __uint_as_float(0x00400000u);
}

__device__ __forceinline__ float e2m1(uint32_t c) {
  const uint32_t e = (c >> 1) & 3u, f = c & 1u;
  const uint32_t bits = e ? ((e + 126u) << 23) | (f << 22) : (f ? (126u << 23) : 0u);
  return __uint_as_float(bits | ((c & 8u) << 28));
}

// The E2M1 code of a value already over its block's scale: the nearest magnitude up to 6, ties to the even code.
__device__ __forceinline__ uint32_t e2m1_code(float v) {
  float m = fabsf(v);
  m = m > 6.f ? 6.f : m;
  uint32_t c = (m > 0.25f) + (m > 0.75f) + (m > 1.25f) + (m > 1.75f) + (m > 2.5f) + (m > 3.5f) + (m > 5.f);
  c += (m == 0.75f) | (m == 1.75f) | (m == 3.5f);
  return c | (v < 0.f ? 8u : 0u);
}

// Two E2M1 codes of a byte (low nibble first) times the block scale, as a BF16 pair (exact: at most 6 bits).
__device__ __forceinline__ uint32_t nvfp4_pair(uint32_t byte, float scale) {
#if CRIT_E2M1_HW
  uint32_t h;
  asm("{\n\t.reg .b8 t0, t1, t2, t3;\n\tmov.b32 {t0, t1, t2, t3}, %1;\n\tcvt.rn.f16x2.e2m1x2 %0, t0;\n\t}"
      : "=r"(h) : "r"(byte));
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  return bf16x2(f.x * scale, f.y * scale);
#else
  return bf16x2(e2m1(byte & 15u) * scale, e2m1(byte >> 4) * scale);
#endif
}

// A word's eight E2M1 codes (element order) times the block scale (`scale` in FP32, `scale2` the same in FP16 on
// both halves), as four BF16 pairs: one unpack and four conversions to FP16, the scale applied in FP16 (exact: at most
// 6 significant bits, 2^-10 to 2688), then widened to BF16 (exact); elsewhere by the bit pattern in FP32.
__device__ __forceinline__ void nvfp4_word(uint32_t word, float scale, uint32_t scale2, uint32_t* out) {
#if CRIT_E2M1_HW
  uint32_t h[4];
  asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\tmov.b32 {b0, b1, b2, b3}, %4;\n\t"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n\tcvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n\tcvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
      : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]) : "r"(word));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __half2 v = __hmul2(*reinterpret_cast<const __half2*>(&h[i]), *reinterpret_cast<const __half2*>(&scale2));
    const float2 f = __half22float2(v);
    out[i] = bf16x2(f.x, f.y);
  }
#else
#pragma unroll
  for (int i = 0; i < 4; ++i) out[i] = nvfp4_pair((word >> (8 * i)) & 0xffu, scale);
#endif
}

// Two E4M3 codes (the low 16 bits, low byte first) as a BF16 pair, times `scale` (a power of two for MXFP8).
template <bool SCALED>
__device__ __forceinline__ uint32_t fp8_pair(uint32_t two, float scale) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two & 0xffffu), __NV_E4M3);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  return SCALED ? bf16x2(f.x * scale, f.y * scale) : bf16x2(f.x, f.y);
}

__device__ __forceinline__ size_t scale_offset(int n, int j, int ks, int ct) {
  if (!ct) return (size_t)n * ks + j;
  const int r = n & 127;
  return ((size_t)(n >> 7) * ct + (j >> 2)) * 512 + (size_t)(r & 31) * 16 + (r >> 5) * 4 + (j & 3);
}

// One matrix row's kLaneBytes contiguous bytes a lane reads a step (the four lanes of a quad read 4 x kLaneBytes of the
// row in a run), in SUBS sub-steps of 32 values: the step from `base` covers [base, base + 128 SUBS), the lane of quad
// position tig its 32 SUBS values from base + 32 SUBS tig, sub-step q the 32 from there + 32 q. pairs(q, out)
// gives sub-step q's 32 values as 16 BF16 pairs: MMA s takes pairs 2 s (values 4 s, + 1) and 2 s + 1 (4 s + 2, + 3). A sub-step past K
// loads nothing and multiplies as zeros (K is a multiple of 32).
template <int WF>
struct Row;

__device__ __forceinline__ uint4 load16(const uint8_t* p, bool valid) {
  return valid ? __ldg(reinterpret_cast<const uint4*>(p)) : make_uint4(0u, 0u, 0u, 0u);
}
__device__ __forceinline__ uint32_t word(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

template <>
struct Row<kNvfp4> {
  static constexpr int SUBS = kLaneBytes / 16;
  uint4 c[SUBS];
  uint32_t sc[SUBS];   // each sub-step's two E4M3 block scales
  __device__ __forceinline__ void load(const Problem& p, int n, int base, int tig) {
    const uint8_t* row = p.w + (size_t)n * (p.K >> 1);
#pragma unroll
    for (int q = 0; q < SUBS; ++q) {
      const int k = base + 32 * (SUBS * tig + q);
      const bool inside = k < p.K;
      c[q] = load16(row + (k >> 1), inside);
      sc[q] = inside ? __ldg(reinterpret_cast<const uint16_t*>(p.scales + scale_offset(n, k >> 4, p.K >> 4, p.ct)))
                     : 0u;
    }
  }
  // Sub-step q's 32 values as 16 BF16 pairs (pair i: values 2i, 2i + 1).
  __device__ __forceinline__ void pairs(int q, uint32_t (&out)[16]) const {
    const float s0 = e4m3(sc[q]), s1 = e4m3(sc[q] >> 8);
    const __half2 h0 = __float2half2_rn(s0), h1 = __float2half2_rn(s1);
    nvfp4_word(c[q].x, s0, *reinterpret_cast<const uint32_t*>(&h0), out);
    nvfp4_word(c[q].y, s0, *reinterpret_cast<const uint32_t*>(&h0), out + 4);
    nvfp4_word(c[q].z, s1, *reinterpret_cast<const uint32_t*>(&h1), out + 8);
    nvfp4_word(c[q].w, s1, *reinterpret_cast<const uint32_t*>(&h1), out + 12);
  }
};

template <>
struct Row<kFp8> {
  static constexpr int SUBS = kLaneBytes / 32;
  uint4 c[2 * SUBS];
  __device__ __forceinline__ void load(const Problem& p, int n, int base, int tig) {
    const uint8_t* row = p.w + (size_t)n * p.K;
#pragma unroll
    for (int q = 0; q < SUBS; ++q) {
      const int k = base + 32 * (SUBS * tig + q);
      c[2 * q] = load16(row + k, k < p.K);
      c[2 * q + 1] = load16(row + k + 16, k < p.K);
    }
  }
  __device__ __forceinline__ void pairs(int q, uint32_t (&out)[16]) const {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t w = word(c[2 * q + (i >> 2)], i & 3);
      out[2 * i] = fp8_pair<false>(w, 1.f);
      out[2 * i + 1] = fp8_pair<false>(w >> 16, 1.f);
    }
  }
};

template <>
struct Row<kMxfp8> {
  static constexpr int SUBS = kLaneBytes / 32;
  uint4 c[2 * SUBS];
  uint32_t mx[SUBS];   // each sub-step's E8M0 exponent
  __device__ __forceinline__ void load(const Problem& p, int n, int base, int tig) {
    const uint8_t* row = p.w + (size_t)n * p.K;
#pragma unroll
    for (int q = 0; q < SUBS; ++q) {
      const int k = base + 32 * (SUBS * tig + q);
      c[2 * q] = load16(row + k, k < p.K);
      c[2 * q + 1] = load16(row + k + 16, k < p.K);
      mx[q] = k < p.K ? __ldg(p.scales + (size_t)n * (p.K >> 5) + (k >> 5)) : 127u;
    }
  }
  __device__ __forceinline__ void pairs(int q, uint32_t (&out)[16]) const {
    const float scale = e8m0(mx[q]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t w = word(c[2 * q + (i >> 2)], i & 3);
      out[2 * i] = fp8_pair<true>(w, scale);
      out[2 * i + 1] = fp8_pair<true>(w >> 16, scale);
    }
  }
};

template <>
struct Row<kBf16> {
  static constexpr int SUBS = kLaneBytes / 64;
  uint4 c[4 * SUBS];
  __device__ __forceinline__ void load(const Problem& p, int n, int base, int tig) {
#pragma unroll
    for (int q = 0; q < SUBS; ++q) {
      const int k = base + 32 * (SUBS * tig + q);
      const uint8_t* at = p.w + ((size_t)n * p.K + k) * 2;
#pragma unroll
      for (int i = 0; i < 4; ++i) c[4 * q + i] = load16(at + 16 * i, k < p.K);
    }
  }
  __device__ __forceinline__ void pairs(int q, uint32_t (&out)[16]) const {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      out[4 * i] = c[4 * q + i].x;
      out[4 * i + 1] = c[4 * q + i].y;
      out[4 * i + 2] = c[4 * q + i].z;
      out[4 * i + 3] = c[4 * q + i].w;
    }
  }
};

// A token's B fragments for a sub-step: its 32 values [kb, kb + 32) as 16 BF16 pairs (word i: values 2i, 2i + 1);
// zeros for a token past M or a sub-step past K.
__device__ __forceinline__ void act_frags(const Problem& p, const __nv_bfloat16* x, int tok, int kb, bool inside,
                                          uint32_t (&b)[16]) {
  if (tok >= p.M || !inside) {
#pragma unroll
    for (int i = 0; i < 16; ++i) b[i] = 0u;
    return;
  }
  const uint4* q = reinterpret_cast<const uint4*>(x + (size_t)tok * p.ldx + kb);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 v = __ldg(q + i);
    b[4 * i] = v.x;
    b[4 * i + 1] = v.y;
    b[4 * i + 2] = v.z;
    b[4 * i + 3] = v.w;
  }
}

__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Programmatic dependent launch (compute capability 9.0 on; nothing below): wait for the previous kernel in the stream
// before reading what it wrote, and let the next one start.
__device__ __forceinline__ void wait_prior() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void launch_next() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// ---- the activations' quantizers (one definition: the pre-passes and the quantize ops) ----

__device__ __forceinline__ float fp8_value(float v, float scale) {
  v = __fdiv_rn(v, scale);
  v = v > 448.f ? 448.f : v < -448.f ? -448.f : v;
  return e4m3(e4m3_bits(v));
}

// A 16-value block (8 BF16 pairs) in NVFP4 under input scale g: its E4M3 scale bits and E2M1 codes (low nibble
// first: element 2i in byte i's low nibble, the 8 bytes in two words).
__device__ __forceinline__ uint32_t nvfp4_block(const uint32_t (&w)[8], float g, float g_rcp, uint32_t& lo, uint32_t& hi) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) amax = nanmax(nanmax(amax, fabsf(bf16_lo(w[i]))), fabsf(bf16_hi(w[i])));
  float bs = (amax * kRcp6) * g;
  bs = bs > 448.f ? 448.f : bs;
  const uint32_t sbits = e4m3_bits(bs);
  float d = e4m3(sbits) * g_rcp;
  const bool zero = !(d > 0.f);
  d = d < 1e-30f ? 1e-30f : d;
  lo = hi = 0u;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t c0 = e2m1_code(zero ? 0.f : __fdiv_rn(bf16_lo(w[i]), d));
    const uint32_t c1 = e2m1_code(zero ? 0.f : __fdiv_rn(bf16_hi(w[i]), d));
    const uint32_t byte = c0 | (c1 << 4);
    if (i < 4) lo |= byte << (8 * i);
    else hi |= byte << (8 * (i - 4));
  }
  return sbits;
}

// One CTA a row: its scale (amax * fp32(1 / 448), at least 1e-12) and its values, as E4M3 bytes (VALUES false) or as
// BF16 values (true: the projection's pre-pass). Rows with K a multiple of 8 on 16-byte boundaries (aligned) read 8
// values a load and keep up to kHeld of them a thread between the maximum and the quantization; others go by value.
constexpr int kRowThreads = 512, kHeld = 4;

template <bool VALUES>
__global__ void __launch_bounds__(kRowThreads) fp8_rows_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, int K,
                                                               void* __restrict__ q, int64_t ldq,
                                                               float* __restrict__ scale, bool aligned) {
  __shared__ float part[kRowThreads / 32];
  __shared__ float row_scale;
  wait_prior();
  launch_next();
  const int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const __nv_bfloat16* xr = x + (size_t)row * ldx;
  float m = 0.f;
  uint4 held[kHeld];
  if (aligned) {
#pragma unroll
    for (int i = 0; i < kHeld; ++i) {
      const int k = 8 * (threadIdx.x + i * kRowThreads);
      held[i] = k < K ? __ldg(reinterpret_cast<const uint4*>(xr + k)) : make_uint4(0u, 0u, 0u, 0u);
      const uint32_t w[4] = {held[i].x, held[i].y, held[i].z, held[i].w};
#pragma unroll
      for (int j = 0; j < 4; ++j) m = nanmax(nanmax(m, fabsf(bf16_lo(w[j]))), fabsf(bf16_hi(w[j])));
    }
    for (int k = 8 * (threadIdx.x + kHeld * kRowThreads); k < K; k += 8 * kRowThreads) {
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(xr + k));
      const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) m = nanmax(nanmax(m, fabsf(bf16_lo(w[j]))), fabsf(bf16_hi(w[j])));
    }
  } else {
    for (int k = threadIdx.x; k < K; k += blockDim.x) m = nanmax(m, fabsf(__bfloat162float(xr[k])));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) m = nanmax(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (lane == 0) part[warp] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) a = nanmax(a, part[w]);
    const float sc = a * kRcp448;
    row_scale = sc < 1e-12f ? 1e-12f : sc;
    scale[row] = row_scale;
  }
  __syncthreads();
  const float sc = row_scale;
  auto put8 = [&](int k, const uint4& v) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    if constexpr (VALUES) {
      uint32_t o[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) o[j] = bf16x2(fp8_value(bf16_lo(w[j]), sc), fp8_value(bf16_hi(w[j]), sc));
      *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(q) + (size_t)row * ldq + k) = make_uint4(o[0], o[1], o[2], o[3]);
    } else {
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t two = e4m3_bits(fp8_value(bf16_lo(w[j]), sc)) | (e4m3_bits(fp8_value(bf16_hi(w[j]), sc)) << 8);
        if (j < 2) lo |= two << (16 * j);
        else hi |= two << (16 * (j - 2));
      }
      *reinterpret_cast<uint2*>(static_cast<uint8_t*>(q) + (size_t)row * ldq + k) = make_uint2(lo, hi);
    }
  };
  if (aligned) {
#pragma unroll
    for (int i = 0; i < kHeld; ++i) {
      const int k = 8 * (threadIdx.x + i * kRowThreads);
      if (k < K) put8(k, held[i]);
    }
    for (int k = 8 * (threadIdx.x + kHeld * kRowThreads); k < K; k += 8 * kRowThreads)
      put8(k, __ldg(reinterpret_cast<const uint4*>(xr + k)));
  } else {
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
      const float v = fp8_value(__bfloat162float(xr[k]), sc);
      if constexpr (VALUES) static_cast<__nv_bfloat16*>(q)[(size_t)row * ldq + k] = __float2bfloat16_rn(v);
      else static_cast<uint8_t*>(q)[(size_t)row * ldq + k] = static_cast<uint8_t>(e4m3_bits(v));
    }
  }
}

// One thread a 16-value block: its values in BF16 (the projection's pre-pass; e2m1(code) * the block's scale).
__global__ void __launch_bounds__(256) nvfp4_values_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, int M,
                                                           int K, float g, float g_rcp,
                                                           __nv_bfloat16* __restrict__ values) {
  wait_prior();
  launch_next();
  const int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int blocks = K >> 4;
  if (id >= (int64_t)M * blocks) return;
  const int row = (int)(id / blocks), j = (int)(id - (int64_t)row * blocks);
  const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)row * ldx + 16 * j);
  const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
  const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
  uint32_t lo, hi;
  const float scale = e4m3(nvfp4_block(w, g, g_rcp, lo, hi));
  uint32_t out[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t byte = ((i < 4 ? lo : hi) >> (8 * (i & 3))) & 0xffu;
    out[i] = bf16x2(e2m1(byte & 15u) * scale, e2m1(byte >> 4) * scale);
  }
  uint4* dst = reinterpret_cast<uint4*>(values + (size_t)row * K + 16 * j);
  dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
  dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
}

// One thread a 16-value block of the padded [ceil(M / 128) * 128][ceil(K / 64) * 4] block grid: its codes and its
// scale, or a zero scale in the padding.
__global__ void __launch_bounds__(256) nvfp4_codes_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, int M,
                                                          int K, float g, float g_rcp, uint8_t* __restrict__ codes,
                                                          uint8_t* __restrict__ scales, int rows_padded,
                                                          int blocks_padded) {
  wait_prior();
  launch_next();
  const int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= (int64_t)rows_padded * blocks_padded) return;
  const int row = (int)(id / blocks_padded), j = (int)(id - (int64_t)row * blocks_padded);
  const size_t at = scale_offset(row, j, 0, blocks_padded >> 2);
  if (row >= M || j >= (K >> 4)) {
    scales[at] = 0;
    return;
  }
  const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)row * ldx + 16 * j);
  const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
  const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
  uint32_t lo, hi;
  scales[at] = static_cast<uint8_t>(nvfp4_block(w, g, g_rcp, lo, hi));
  *reinterpret_cast<uint2*>(codes + (size_t)row * (K >> 1) + 8 * j) = make_uint2(lo, hi);
}

// ---- the projection ----


// A finished sum: its scales applied in a fixed order, then stored in out's type.
template <int WF, int ACT>
__device__ __forceinline__ void finish(const Problem& p, int tok, int n, float v) {
  if constexpr (WF == kNvfp4 || WF == kFp8) v *= p.row_scale[n];
  if constexpr (ACT == kActNvfp4) v *= p.input_rcp;
  if constexpr (ACT == kActFp8Rows) v *= p.act_scale[tok];
  const size_t at = (size_t)tok * p.ldo + n;
  if (p.out_kind == kOutF32) static_cast<float*>(p.out)[at] = v;
  else if (p.out_kind == kOutBf16) static_cast<__nv_bfloat16*>(p.out)[at] = __float2bfloat16_rn(v);
  else static_cast<__half*>(p.out)[at] = __float2half_rn(v);
}

// One CTA: a row tile (warps x 16 R matrix rows, each warp its own) times every token (up to 8 T) over split
// blockIdx.y of K's steps.
template <int WF, int ACT, int R, int T>
// Decode tiles (up to 16 rows) at most 128 registers a thread, so four CTAs of four warps fit a multiprocessor (a few
// percent faster than the compiler's own choice at 1-4 rows); prompt tiles as the compiler likes.
__global__ void __launch_bounds__(kMaxWarps * 32, T <= 2 ? 4 : 1) mma_kernel(const Problem p) {
  constexpr int SUBS = Row<WF>::SUBS, STEP = 128 * SUBS;
  __shared__ int last;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int tile = blockIdx.x, split = blockIdx.y, rows = tile_rows(R, p.warps);
  const int r0 = tile * rows + warp * 16 * R;
  const int steps = (p.K + STEP - 1) / STEP;
  const int s_lo = split * steps / p.splits, s_hi = (split + 1) * steps / p.splits;
  const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(p.x);

  // One accumulator chain a tile (two chains, even and odd MMAs, measured slower: more registers, fewer warps).
  float c[R][T][4];
#pragma unroll
  for (int g = 0; g < R; ++g)
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) c[g][t][i] = 0.f;

  // One step's weights in registers; the warps a multiprocessor holds keep the loads in flight (a second set of
  // registers for the next step measured slower: fewer warps fit).
  Row<WF> wa[R], wb[R];
  auto load = [&](int st) {
#pragma unroll
    for (int g = 0; g < R; ++g) {
      wa[g].load(p, min(r0 + 16 * g + gid, p.N - 1), st * STEP, tig);   // a row past N reads row N - 1, writes nothing
      wb[g].load(p, min(r0 + 16 * g + gid + 8, p.N - 1), st * STEP, tig);
    }
  };
  if (s_lo < s_hi) load(s_lo);   // weights do not depend on the previous kernel: read them before waiting for it
  wait_prior();
  for (int st = s_lo; st < s_hi;) {
#pragma unroll
    for (int q = 0; q < SUBS; ++q) {
      const int kb = st * STEP + 32 * (SUBS * tig + q);
      uint32_t b[T][16];
#pragma unroll
      for (int t = 0; t < T; ++t) act_frags(p, x, 8 * t + gid, kb, kb < p.K, b[t]);
#pragma unroll
      for (int g = 0; g < R; ++g) {
        uint32_t pa[16], pb[16];   // MMA s takes pairs 2 s (values 4 s, + 1) and 2 s + 1 of rows gid and gid + 8
        wa[g].pairs(q, pa);
        wb[g].pairs(q, pb);
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
          for (int t = 0; t < T; ++t)
            mma(c[g][t], pa[2 * s], pb[2 * s], pa[2 * s + 1], pb[2 * s + 1], b[t][2 * s], b[t][2 * s + 1]);
      }
    }
    if (++st < s_hi) load(st);
  }

  // c0, c1: matrix row gid, tokens 2 tig and 2 tig + 1; c2, c3: matrix row gid + 8.
  if (p.splits == 1) {
#pragma unroll
    for (int g = 0; g < R; ++g)
#pragma unroll
      for (int t = 0; t < T; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int n = r0 + 16 * g + gid + (i >> 1) * 8, tok = 8 * t + 2 * tig + (i & 1);
          if (n < p.N && tok < p.M) finish<WF, ACT>(p, tok, n, c[g][t][i]);
        }
    return;
  }
  float* part = p.partial + (size_t)split * p.M * p.N;
#pragma unroll
  for (int g = 0; g < R; ++g)
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = r0 + 16 * g + gid + (i >> 1) * 8, tok = 8 * t + 2 * tig + (i & 1);
        if (n < p.N && tok < p.M) __stcg(part + (size_t)tok * p.N + n, c[g][t][i]);
      }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(p.counters + tile, 1) == p.splits - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (threadIdx.x == 0) p.counters[tile] = 0;   // every split has counted: ready for the next call
  const int first = tile * rows;
  for (int i = threadIdx.x; i < p.M * rows; i += blockDim.x) {
    const int tok = i / rows, n = first + (i - tok * rows);
    if (n >= p.N) continue;
    float v = __ldcg(p.partial + (size_t)tok * p.N + n);
    for (int s = 1; s < p.splits; ++s) v += __ldcg(p.partial + ((size_t)s * p.M + tok) * p.N + n);
    finish<WF, ACT>(p, tok, n, v);
  }
}

bool programmatic(cudaStream_t stream) {
  static int major[64] = {};
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) return false;
  if (!major[device] && cudaDeviceGetAttribute(&major[device], cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
    return false;
  return major[device] >= 9;
}

template <class... Params, class... Args>
cudaError_t launch_kernel(void (*kernel)(Params...), dim3 grid, dim3 block, cudaStream_t stream, Args... args) {
  cudaLaunchAttribute attribute[1];
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.stream = stream;
  if (programmatic(stream)) {
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attribute;
    config.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&config, kernel, args...);
}

template <int WF, int ACT, int R, int T>
cudaError_t launch(const Problem& p, cudaStream_t stream) {
  const dim3 grid(row_tiles(p.N, R, p.warps), p.splits);
  return launch_kernel(mma_kernel<WF, ACT, R, T>, grid, dim3(32 * p.warps), stream, p);
}

// R * T <= 8 keeps a lane's accumulators at 32 registers.
template <int WF, int ACT>
cudaError_t by_tile(const Problem& p, cudaStream_t stream) {
  const int T = p.M <= 8 ? 1 : p.M <= 16 ? 2 : p.M <= 32 ? 4 : 8;
  const int R = T <= 4 ? p.groups : 1;
  if (T == 1) return R == 2 ? launch<WF, ACT, 2, 1>(p, stream) : launch<WF, ACT, 1, 1>(p, stream);
  if (T == 2) return R == 2 ? launch<WF, ACT, 2, 2>(p, stream) : launch<WF, ACT, 1, 2>(p, stream);
  if (T == 4) return R == 2 ? launch<WF, ACT, 2, 4>(p, stream) : launch<WF, ACT, 1, 4>(p, stream);
  return launch<WF, ACT, 1, 8>(p, stream);
}

}  // namespace

cudaError_t enqueue(const Problem& given, cudaStream_t stream) {
  Problem p = given;
  if (p.M < 1 || p.M > kMaxRows || p.K < kStep || p.K % kStep || p.N < 1 || p.warps < 1 || p.warps > kMaxWarps ||
      (p.groups != 1 && p.groups != 2) || p.splits < 1 || p.splits > kMaxSplits ||
      (p.splits > 1 && (!p.partial || !p.counters)))
    return cudaErrorInvalidValue;
  // A row tile's rows (a quad of a tile's warps reads its 16) are rows of one tile: the counters are a tile's.
  if (p.acts == kActNvfp4Mma) return enqueue_fp4(p, stream);   // fp4_mma.cu
  if (p.acts != kActBf16) {
    if (!p.scratch) return cudaErrorInvalidValue;
    __nv_bfloat16* values = reinterpret_cast<__nv_bfloat16*>(p.scratch);
    float* scales = reinterpret_cast<float*>(p.scratch + ((size_t)p.M * p.K * 2 + 15) / 16 * 16);
    cudaError_t e;
    if (p.acts == kActFp8Rows) {
      e = launch_kernel(fp8_rows_kernel<true>, dim3(p.M), dim3(kRowThreads), stream,
                        static_cast<const __nv_bfloat16*>(p.x), p.ldx, p.K, static_cast<void*>(values), (int64_t)p.K,
                        scales, p.K % 8 == 0 && p.ldx % 8 == 0);
    } else {
      const int64_t threads = (int64_t)p.M * (p.K >> 4);
      e = launch_kernel(nvfp4_values_kernel, dim3((unsigned)((threads + 255) / 256)), dim3(256), stream,
                        static_cast<const __nv_bfloat16*>(p.x), p.ldx, p.M, p.K, p.input_scale, p.input_rcp, values);
    }
    if (e != cudaSuccess) return e;
    p.x = values;
    p.ldx = p.K;
    p.act_scale = scales;
  }
  if (p.weights == kNvfp4 && p.acts == kActBf16) return by_tile<kNvfp4, kActBf16>(p, stream);
  if (p.weights == kNvfp4 && p.acts == kActNvfp4) return by_tile<kNvfp4, kActNvfp4>(p, stream);
  if (p.weights == kFp8 && p.acts == kActBf16) return by_tile<kFp8, kActBf16>(p, stream);
  if (p.weights == kFp8 && p.acts == kActFp8Rows) return by_tile<kFp8, kActFp8Rows>(p, stream);
  if (p.weights == kMxfp8 && p.acts == kActBf16) return by_tile<kMxfp8, kActBf16>(p, stream);
  if (p.weights == kBf16 && p.acts == kActBf16) return by_tile<kBf16, kActBf16>(p, stream);
  return cudaErrorInvalidValue;
}

cudaError_t enqueue_quantize_fp8_rows(const void* x, int64_t ldx, int M, int K, std::uint8_t* q, int64_t ldq,
                                      float* scale, cudaStream_t stream) {
  if (M < 1 || K < 1) return cudaErrorInvalidValue;
  const bool aligned = K % 8 == 0 && ldx % 8 == 0 && ldq % 8 == 0 && reinterpret_cast<std::uintptr_t>(x) % 16 == 0 &&
                       reinterpret_cast<std::uintptr_t>(q) % 8 == 0;
  return launch_kernel(fp8_rows_kernel<false>, dim3(M), dim3(kRowThreads), stream, static_cast<const __nv_bfloat16*>(x),
                       ldx, K, static_cast<void*>(q), ldq, scale, aligned);
}

cudaError_t enqueue_quantize_nvfp4(const void* x, int64_t ldx, int M, int K, float input_scale, float input_rcp,
                                   std::uint8_t* codes, std::uint8_t* blocked_scales, cudaStream_t stream) {
  if (M < 1 || K < 16 || K % 16) return cudaErrorInvalidValue;
  const int rows_padded = (M + 127) / 128 * 128, blocks_padded = (K / 16 + 3) / 4 * 4;
  const int64_t threads = (int64_t)rows_padded * blocks_padded;
  return launch_kernel(nvfp4_codes_kernel, dim3((unsigned)((threads + 255) / 256)), dim3(256), stream,
                       static_cast<const __nv_bfloat16*>(x), ldx, M, K, input_scale, input_rcp, codes, blocked_scales,
                       rows_padded, blocks_padded);
}

}  // namespace crit::kernels::mma_rows
