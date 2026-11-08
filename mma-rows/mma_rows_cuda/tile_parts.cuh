// The device parts mma-rows' two tiles share (mma_rows.cu's decode tile for 1 to 64 rows, prompt_tile.cu's prompt
// tile for any number of rows): the value decoders, a step's weights in registers (Row), the activations' B
// fragments, the MMA, programmatic dependent launch, the quantizers' value functions and a finished sum's scales.
// One definition, so both tiles feed every MMA the same operands and finish every sum with the same instructions.
#pragma once
#include "mma_rows.h"

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
  // Word w (0..3) of sub-step q: its 8 values as BF16 pairs 4w .. 4w + 3 of the sub-step (MMAs 2w and 2w + 1).
  __device__ __forceinline__ void quad(int q, int w, uint32_t* out) const {
    const float s = e4m3(sc[q] >> (8 * (w >> 1)));
    const __half2 h = __float2half2_rn(s);
    nvfp4_word(word(c[q], w), s, *reinterpret_cast<const uint32_t*>(&h), out);
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
  __device__ __forceinline__ void quad(int q, int w, uint32_t* out) const {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int i = 2 * w + j;
      const uint32_t v = word(c[2 * q + (i >> 2)], i & 3);
      out[2 * j] = fp8_pair<false>(v, 1.f);
      out[2 * j + 1] = fp8_pair<false>(v >> 16, 1.f);
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
  __device__ __forceinline__ void quad(int q, int w, uint32_t* out) const {
    const float scale = e8m0(mx[q]);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int i = 2 * w + j;
      const uint32_t v = word(c[2 * q + (i >> 2)], i & 3);
      out[2 * j] = fp8_pair<true>(v, scale);
      out[2 * j + 1] = fp8_pair<true>(v >> 16, scale);
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
  __device__ __forceinline__ void quad(int q, int w, uint32_t* out) const {
    out[0] = c[4 * q + w].x;
    out[1] = c[4 * q + w].y;
    out[2] = c[4 * q + w].z;
    out[3] = c[4 * q + w].w;
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

// A launch on `stream`, as the programmatic dependent of the kernel before it where the device has it (9.0 on).
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

}  // namespace
}  // namespace crit::kernels::mma_rows
