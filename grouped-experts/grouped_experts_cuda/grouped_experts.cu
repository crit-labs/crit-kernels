// Routed SwiGLU experts over bundles (grouped_experts.h). BF16 mma.sync m16n8k16 (sm_80 and newer) for every weight
// format: a warp owns 16 output rows of one expert and every route of its item, so each output is the same chain of
// MMAs whether its item holds 1 route or 32, and decode rows and prompt chunks compute the same function.
#include "grouped_experts.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace crit::techniques::grouped_experts {
namespace {

using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

constexpr int kPad = 8;   // BF16 elements of padding per staged route row (16 bytes): conflict-free fragment loads

__device__ __forceinline__ uint32_t u32(bf162 v) { return *reinterpret_cast<const uint32_t*>(&v); }
__device__ __forceinline__ bf162 b2(uint32_t v) { return *reinterpret_cast<const bf162*>(&v); }

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ------------------------------------------------------------------------------------------------ exact decoders

// An E4M3 byte's value: its bits as FP16 are the value times 2^-8, subnormals included (both formats' subnormals
// have exponent field 0), so the conversion is exact. Byte 0x7F (NaN) reads as 480: scales never hold it.
__device__ __forceinline__ float e4m3_value(uint32_t b) {
    const unsigned short h = static_cast<unsigned short>(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8));
    return __half2float(__ushort_as_half(h)) * 256.0f;
}

// Eight E2M1 codes of a word (element 2i in the low nibble of byte i) as four BF16 pairs, exactly, by two byte
// permutations per four elements: the magnitude's high and low BF16 bytes from 8-entry tables, the sign from bit 3.
__device__ __forceinline__ void e2m1x8(uint32_t w, uint32_t* out) {
    constexpr uint32_t kHiA = 0x3F3F3F00u, kHiB = 0x40404040u, kLoA = 0xC0800000u, kLoB = 0xC0804000u;
#pragma unroll
    for(int half = 0; half < 2; ++half) {
        const uint32_t v = w >> (16 * half), sel = v & 0x7777u;
        const uint32_t hi = __byte_perm(kHiA, kHiB, sel), lo = __byte_perm(kLoA, kLoB, sel);
        out[2 * half] = __byte_perm(lo, hi, 0x5140) | ((v << 12) & 0x8000u) | ((v << 24) & 0x80000000u);
        out[2 * half + 1] = __byte_perm(lo, hi, 0x7362) | ((v << 4) & 0x8000u) | ((v << 16) & 0x80000000u);
    }
}

// Eight INT4 codes (two's complement nibbles) times a BF16 scale: (128 + (code ^ 8)) is exact in BF16, and
// fma(that, s, -136 s) = (code - 8 + 8 ^ ...) * s = value * s exactly (the product needs at most 7 significant bits).
__device__ __forceinline__ void int4x8(uint32_t w, bf162 s, uint32_t* out) {
    const uint32_t x = w ^ 0x88888888u;
    const bf162 bias = __hmul2(s, __floats2bfloat162_rn(-136.0f, -136.0f));
#pragma unroll
    for(int p = 0; p < 4; ++p) {
        const uint32_t byte = (x >> (8 * p)) & 0xFFu;
        const uint32_t word = 0x43004300u | (byte & 0xFu) | ((byte & 0xF0u) << 12);
        out[p] = u32(__hfma2(b2(word), s, bias));
    }
}

// Four E4M3 bytes as two BF16 pairs, exactly.
__device__ __forceinline__ void e4m3x4(uint32_t w, uint32_t* out) {
    float f[4];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w & 0xFFFFu), __NV_E4M3);
    const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w >> 16), __NV_E4M3);
    const float2 a = __half22float2(__half2(lo)), b = __half22float2(__half2(hi));
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
#else
#pragma unroll
    for(int i = 0; i < 4; ++i) f[i] = e4m3_value((w >> (8 * i)) & 0xFFu);
#endif
    out[0] = u32(__floats2bfloat162_rn(f[0], f[1]));
    out[1] = u32(__floats2bfloat162_rn(f[2], f[3]));
}

// ------------------------------------------------------------------------------------------- declared operations

__device__ __forceinline__ float exp_declared(float x) {
    x = x > 88.0f ? 88.0f : x < -87.0f ? -87.0f : x;
    const float n = rintf(__fmul_rn(x, 1.44269504f));
    float r = __fmaf_rn(n, -0.693145752f, x);
    r = __fmaf_rn(n, -1.42860677e-06f, r);
    float p = 1.98412698e-04f;
    p = __fmaf_rn(p, r, 1.38888889e-03f);
    p = __fmaf_rn(p, r, 8.33333333e-03f);
    p = __fmaf_rn(p, r, 4.16666667e-02f);
    p = __fmaf_rn(p, r, 1.66666667e-01f);
    p = __fmaf_rn(p, r, 0.5f);
    p = __fmaf_rn(p, r, 1.0f);
    p = __fmaf_rn(p, r, 1.0f);
    return __fmul_rn(p, __int_as_float((int(n) + 127) << 23));
}

__device__ __forceinline__ float swiglu_declared(float gate, float up) {
    return __fmul_rn(__fdiv_rn(gate, __fadd_rn(1.0f, exp_declared(-gate))), up);
}

// The nearest E4M3 code of a non-negative finite x, ties to the even code, saturating at 448 (crit::nvfp4's).
__device__ __forceinline__ uint32_t e4m3_encode(float x) {
    x = fminf(x, 448.0f);
    if(x < 0.015625f) return static_cast<uint32_t>(__float2int_rn(__fmul_rn(x, 512.0f)));   // the subnormal grid
    const uint32_t bits = __float_as_uint(x);
    const uint32_t keep = (bits >> 20) & 7u, rest = bits & 0xFFFFFu;
    uint32_t code = ((((bits >> 23) & 0xFFu) - 120u) << 3) | keep;
    if(rest > 0x80000u || (rest == 0x80000u && (keep & 1u))) ++code;
    return code > 0x7Eu ? 0x7Eu : code;
}

// The nearest E2M1 magnitude code of a >= 0, ties to the even code (NaN: 0).
__device__ __forceinline__ int e2m1_code(float a) {
    return int(a > 0.25f) + int(a >= 0.75f) + int(a > 1.25f) + int(a >= 1.75f) + int(a > 2.5f) + int(a >= 3.5f) +
           int(a > 5.0f);
}
__device__ __forceinline__ float e2m1_magnitude(int code) {
    return code < 4 ? 0.5f * float(code) : code < 6 ? float(code - 2) : code == 6 ? 4.0f : 6.0f;
}

// The declared quantizer's block scale (decoded) for a block of maximum magnitude amax at global g.
__device__ __forceinline__ float block_scale(float amax, float g) {
    float wanted = __fdiv_rn(__fmul_rn(amax, g), 6.0f);
    if(!isfinite(wanted)) wanted = 448.0f;
    return e4m3_value(e4m3_encode(wanted));
}
// One value's E2M1 operand under its block's decoded scale: e2m1(code) * decoded, the code of (v * g) / decoded.
__device__ __forceinline__ float e2m1_operand(float v, float g, float decoded) {
    if(!(decoded > 0.0f)) return 0.0f;
    const float q = __fdiv_rn(__fmul_rn(v, g), decoded);
    const int code = e2m1_code(fabsf(q));
    const float m = __fmul_rn(e2m1_magnitude(code), decoded);
    return (q < 0.0f && code) ? -m : m;
}

// ----------------------------------------------------------------------------------------------- weight tiles

template <int F>
struct Raw {
    uint32_t w[F == int(Format::bf16) ? 8 : (F == int(Format::fp8_channel) || F == int(Format::fp8_block)) ? 4 : 2];
    uint32_t s;
};

template <int F>
constexpr int depth() {   // 64-k blocks of weights in flight per warp
    return F == int(Format::bf16) ? 2 : (F == int(Format::fp8_channel) || F == int(Format::fp8_block)) ? 3 : 4;
}

// The lane's 16 weights k .. k + 15 of `row` (k a multiple of 16).
template <int F>
__device__ __forceinline__ Raw<F> load_raw(const std::uint8_t* codes, const std::uint8_t* scales, int row, int K,
                                           int k) {
    Raw<F> r;
    if constexpr(F == int(Format::bf16)) {
        const uint4* p = reinterpret_cast<const uint4*>(codes + (static_cast<long long>(row) * K + k) * 2);
        const uint4 a = __ldg(p), b = __ldg(p + 1);
        r.w[0] = a.x; r.w[1] = a.y; r.w[2] = a.z; r.w[3] = a.w;
        r.w[4] = b.x; r.w[5] = b.y; r.w[6] = b.z; r.w[7] = b.w;
        r.s = 0;
    } else if constexpr(F == int(Format::fp8_channel) || F == int(Format::fp8_block)) {
        const uint4 a = __ldg(reinterpret_cast<const uint4*>(codes + static_cast<long long>(row) * K + k));
        r.w[0] = a.x; r.w[1] = a.y; r.w[2] = a.z; r.w[3] = a.w;
        r.s = 0;
    } else {
        const uint2 a = __ldg(reinterpret_cast<const uint2*>(codes + static_cast<long long>(row) * (K / 2) + k / 2));
        r.w[0] = a.x; r.w[1] = a.y;
        if constexpr(F == int(Format::mxfp4)) r.s = __ldg(scales + static_cast<long long>(row) * (K / 32) + k / 32);
        else r.s = __ldg(scales + static_cast<long long>(row) * (K / 16) + k / 16);
    }
    return r;
}

// Raw weights as 8 BF16 pairs (element order), exact.
template <int F>
__device__ __forceinline__ void decode(const Raw<F>& r, uint32_t* f) {
    if constexpr(F == int(Format::bf16)) {
#pragma unroll
        for(int i = 0; i < 8; ++i) f[i] = r.w[i];
    } else if constexpr(F == int(Format::fp8_channel) || F == int(Format::fp8_block)) {
#pragma unroll
        for(int i = 0; i < 4; ++i) e4m3x4(r.w[i], f + 2 * i);
    } else if constexpr(F == int(Format::int4)) {
        const bf16 s = __float2bfloat16_rn(e4m3_value(r.s));
        const bf162 s2 = __halves2bfloat162(s, s);
        int4x8(r.w[0], s2, f);
        int4x8(r.w[1], s2, f + 4);
    } else {
        bf162 s2;
        if constexpr(F == int(Format::mxfp4)) {
            const uint32_t bits = (r.s & 0xFFu) << 7;
            s2 = b2(bits | (bits << 16));
        } else {
            const bf16 s = __float2bfloat16_rn(e4m3_value(r.s));
            s2 = __halves2bfloat162(s, s);
        }
        e2m1x8(r.w[0], f);
        e2m1x8(r.w[1], f + 4);
#pragma unroll
        for(int i = 0; i < 8; ++i) f[i] = u32(__hmul2(b2(f[i]), s2));
    }
}

// ----------------------------------------------------------------------------------------------- the expert pass

struct Pass {
    const std::uint8_t* store;
    Layout layout;
    const int* item_slot;
    const int* item_routes;
    const int* item_count;
    const bf16* in;      // gate/up: x [tokens][hidden]; down: h [routes][inter]
    int in_div;          // a route's input row is route / in_div (top for gate/up, 1 for down)
    void* out;           // gate/up: h BF16 [routes][inter]; down: y FP32 [routes][hidden]
    int chunk;           // k values staged per pass (a multiple of 64)
};

template <int F, bool A4, bool GATEUP>
__device__ __forceinline__ float row_factor(const std::uint8_t* bundle, const Layout& L, int row) {
    if constexpr(F == int(Format::fp8_channel)) {
        return reinterpret_cast<const float*>(bundle + (GATEUP ? L.gate_up_scale : L.down_scale))[row];
    } else if constexpr(F == int(Format::nvfp4) || F == int(Format::int4)) {
        float f = reinterpret_cast<const float*>(bundle + (GATEUP ? L.gate_up_inverse : L.down_inverse))[row];
        if constexpr(A4) f = __fmul_rn(f, reinterpret_cast<const float*>(bundle + L.input)[GATEUP ? 2 : 3]);
        return f;
    } else {
        return 1.0f;
    }
}

// Stage the item's inputs k0 .. k0 + count into shared memory, [route][chunk + pad] BF16; a4 quantizes x per 16.
template <bool QUANTIZE>
__device__ __forceinline__ void stage(const Pass& p, const int* routes, int rows, int K, int k0, int count, float g,
                                      bf16* tile) {
    const int width = p.chunk + kPad;
    if constexpr(!QUANTIZE) {
        const int per = count / 8;
        for(int i = threadIdx.x; i < rows * per; i += blockDim.x) {
            const int j = i / per, c = (i % per) * 8, r = routes[j];
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if(r >= 0) v = __ldg(reinterpret_cast<const uint4*>(p.in + static_cast<long long>(r / p.in_div) * K + k0 + c));
            *reinterpret_cast<uint4*>(tile + j * width + c) = v;
        }
    } else {
        const int per = count / 16;
        for(int i = threadIdx.x; i < rows * per; i += blockDim.x) {
            const int j = i / per, c = (i % per) * 16, r = routes[j];
            uint4 out[2] = {make_uint4(0u, 0u, 0u, 0u), make_uint4(0u, 0u, 0u, 0u)};
            if(r >= 0) {
                const uint4* src = reinterpret_cast<const uint4*>(p.in + static_cast<long long>(r / p.in_div) * K + k0 + c);
                const uint4 a = __ldg(src), b = __ldg(src + 1);
                const uint32_t words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                float v[16], amax = 0.0f;
#pragma unroll
                for(int q = 0; q < 8; ++q) {
                    const float2 f = __bfloat1622float2(b2(words[q]));
                    v[2 * q] = f.x;
                    v[2 * q + 1] = f.y;
                    amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
                }
                const float decoded = block_scale(amax, g);
                uint32_t* o = reinterpret_cast<uint32_t*>(out);
#pragma unroll
                for(int q = 0; q < 8; ++q)
                    o[q] = u32(__floats2bfloat162_rn(e2m1_operand(v[2 * q], g, decoded),
                                                     e2m1_operand(v[2 * q + 1], g, decoded)));
            }
            uint4* dst = reinterpret_cast<uint4*>(tile + j * width + c);
            dst[0] = out[0];
            dst[1] = out[1];
        }
    }
}

// One block: item blockIdx.y, WARPS warps of 16 * TILES output rows each (gate/up: TILES 2, 16 inter rows, each MMA
// tile 8 gate rows over the same 8 up rows so a lane holds both halves of its SwiGLU).
template <int F, int NT, int TILES, int WARPS, bool GATEUP, bool A4>
__global__ void __launch_bounds__(WARPS * 32) expert_kernel(Pass p) {
    constexpr int R = 8 * NT, ROWS = 2 * TILES, D = depth<F>();
    constexpr bool kBlock = F == int(Format::fp8_block);
    extern __shared__ __align__(16) unsigned char smem[];
    bf16* tile = reinterpret_cast<bf16*>(smem);
    __shared__ int routes[R];
    const int u = blockIdx.y;
    if(u >= *p.item_count) return;
    for(int j = threadIdx.x; j < R; j += blockDim.x) routes[j] = p.item_routes[static_cast<long long>(u) * R + j];
    __syncthreads();
    int active = 0;
#pragma unroll
    for(int n = 0; n < NT; ++n)
        if(routes[8 * n] >= 0) active = n + 1;
    const Layout& L = p.layout;
    const std::uint8_t* bundle = p.store + static_cast<long long>(p.item_slot[u]) * L.bytes;
    const int H = L.hidden, I = L.inter;
    const int K = GATEUP ? H : I, M = GATEUP ? I : H;
    const std::uint8_t* codes = bundle + (GATEUP ? L.gate_up : L.down);
    const std::uint8_t* scales = bundle + (GATEUP ? L.gate_up_scale : L.down_scale);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int r0 = (blockIdx.x * WARPS + warp) * 16 * (GATEUP ? 1 : TILES);
    const bool live = r0 < M;
    int rows[ROWS];
#pragma unroll
    for(int t = 0; t < TILES; ++t) {
        if constexpr(GATEUP) {
            rows[2 * t] = r0 + 8 * t + g;
            rows[2 * t + 1] = I + r0 + 8 * t + g;
        } else {
            rows[2 * t] = r0 + 16 * t + g;
            rows[2 * t + 1] = r0 + 16 * t + 8 + g;
        }
    }
    const float g_in = A4 && GATEUP ? reinterpret_cast<const float*>(bundle + L.input)[0] : 0.0f;
    float acc[TILES][NT][4], part[kBlock ? TILES : 1][kBlock ? NT : 1][4];
#pragma unroll
    for(int t = 0; t < TILES; ++t)
#pragma unroll
        for(int n = 0; n < NT; ++n)
#pragma unroll
            for(int i = 0; i < 4; ++i) acc[t][n][i] = 0.0f;
    if constexpr(kBlock) {
#pragma unroll
        for(int t = 0; t < TILES; ++t)
#pragma unroll
            for(int n = 0; n < NT; ++n)
#pragma unroll
                for(int i = 0; i < 4; ++i) part[t][n][i] = 0.0f;
    }
    const int blocks = K / 64, width = p.chunk + kPad;
    Raw<F> ring[D][ROWS];
    if(live) {
#pragma unroll
        for(int d = 0; d < D; ++d)
            if(d < blocks)
#pragma unroll
                for(int r = 0; r < ROWS; ++r) ring[d][r] = load_raw<F>(codes, scales, rows[r], K, 64 * d + 16 * t4);
    }
    for(int b0 = 0; b0 < blocks; b0 += D) {
#pragma unroll
        for(int d = 0; d < D; ++d) {
            const int kb = b0 + d;
            if(kb >= blocks) break;
            const int k = 64 * kb, kk = k % p.chunk;
            if(kk == 0) {
                __syncthreads();
                const int count = K - k < p.chunk ? K - k : p.chunk;
                stage<A4 && GATEUP>(p, routes, 8 * active, K, k, count, g_in, tile);
                __syncthreads();
            }
            if(!live) continue;
            uint32_t fr[ROWS][8];
#pragma unroll
            for(int r = 0; r < ROWS; ++r) decode<F>(ring[d][r], fr[r]);
            if(kb + D < blocks) {
#pragma unroll
                for(int r = 0; r < ROWS; ++r) ring[d][r] = load_raw<F>(codes, scales, rows[r], K, 64 * (kb + D) + 16 * t4);
            }
#pragma unroll
            for(int n = 0; n < NT; ++n) {
                if(n >= active) break;
                const uint4* bp = reinterpret_cast<const uint4*>(tile + (8 * n + g) * width + kk + 16 * t4);
                const uint4 v0 = bp[0], v1 = bp[1];
                const uint32_t bw[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                for(int t = 0; t < TILES; ++t)
#pragma unroll
                    for(int s = 0; s < 4; ++s)
                        mma(kBlock ? part[kBlock ? t : 0][kBlock ? n : 0] : acc[t][n], fr[2 * t][2 * s],
                            fr[2 * t + 1][2 * s], fr[2 * t][2 * s + 1], fr[2 * t + 1][2 * s + 1], bw[2 * s],
                            bw[2 * s + 1]);
            }
            if constexpr(kBlock) {
                if((kb & 1) || kb + 1 == blocks) {   // the end of a 128-k scale block
                    const int kblocks = (K + 127) / 128;
                    const float* bs = reinterpret_cast<const float*>(scales);
#pragma unroll
                    for(int t = 0; t < TILES; ++t) {
                        const float s_lo = bs[(rows[2 * t] / 128) * kblocks + kb / 2];
                        const float s_hi = bs[(rows[2 * t + 1] / 128) * kblocks + kb / 2];
#pragma unroll
                        for(int n = 0; n < NT; ++n) {
                            if(n >= active) break;
#pragma unroll
                            for(int i = 0; i < 4; ++i) {
                                acc[t][n][i] = __fmaf_rn(part[t][n][i], i < 2 ? s_lo : s_hi, acc[t][n][i]);
                                part[t][n][i] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }
    if(!live) return;
    if constexpr(GATEUP) {
        // C fragment: c0, c1 = gate row rows[2t] for routes 2 t4, 2 t4 + 1; c2, c3 = up row rows[2t + 1], same routes.
        bf16* h = reinterpret_cast<bf16*>(p.out);
        float fg[TILES], fu[TILES];
#pragma unroll
        for(int t = 0; t < TILES; ++t) {
            fg[t] = row_factor<F, A4, true>(bundle, L, rows[2 * t]);
            fu[t] = row_factor<F, A4, true>(bundle, L, rows[2 * t + 1]);
        }
        const float g_dn = A4 ? reinterpret_cast<const float*>(bundle + L.input)[1] : 0.0f;
#pragma unroll
        for(int n = 0; n < NT; ++n) {
            if(n >= active) break;
#pragma unroll
            for(int c = 0; c < 2; ++c) {
                float hv[TILES];
#pragma unroll
                for(int t = 0; t < TILES; ++t)
                    hv[t] = swiglu_declared(__fmul_rn(acc[t][n][c], fg[t]), __fmul_rn(acc[t][n][2 + c], fu[t]));
                if constexpr(A4) {   // the 16 inter rows r0 .. r0 + 15 of this route: one block of the down's input
                    float amax = 0.0f;
#pragma unroll
                    for(int t = 0; t < TILES; ++t) amax = fmaxf(amax, fabsf(hv[t]));
#pragma unroll
                    for(int o = 4; o < 32; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                    const float decoded = block_scale(amax, g_dn);
#pragma unroll
                    for(int t = 0; t < TILES; ++t) hv[t] = e2m1_operand(hv[t], g_dn, decoded);
                }
                const int r = routes[8 * n + 2 * t4 + c];
                if(r >= 0) {
#pragma unroll
                    for(int t = 0; t < TILES; ++t)
                        h[static_cast<long long>(r) * I + r0 + 8 * t + g] = __float2bfloat16_rn(hv[t]);
                }
            }
        }
    } else {
        float* y = reinterpret_cast<float*>(p.out);
        float f[ROWS];
#pragma unroll
        for(int r = 0; r < ROWS; ++r) f[r] = row_factor<F, A4, false>(bundle, L, rows[r]);
#pragma unroll
        for(int n = 0; n < NT; ++n) {
            if(n >= active) break;
#pragma unroll
            for(int t = 0; t < TILES; ++t)
#pragma unroll
                for(int i = 0; i < 4; ++i) {
                    const int r = routes[8 * n + 2 * t4 + (i & 1)];
                    const int row = rows[2 * t + (i >> 1)];
                    if(r >= 0) y[static_cast<long long>(r) * H + row] = __fmul_rn(acc[t][n][i], f[2 * t + (i >> 1)]);
                }
        }
    }
}

// ----------------------------------------------------------------------------------------------- plans

template <bool MAX>
__device__ __forceinline__ int block_scan(int v, int* buffer) {   // inclusive, over blockDim.x (a multiple of 32)
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
#pragma unroll
    for(int o = 1; o < 32; o <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, v, o);
        if(lane >= o) v = MAX ? max(v, other) : v + other;
    }
    if(lane == 31) buffer[warp] = v;
    __syncthreads();
    if(warp == 0) {
        int w = lane < warps ? buffer[lane] : (MAX ? 0 : 0);
#pragma unroll
        for(int o = 1; o < 32; o <<= 1) {
            const int other = __shfl_up_sync(0xffffffffu, w, o);
            if(lane >= o) w = MAX ? max(w, other) : w + other;
        }
        if(lane < warps) buffer[lane] = w;
    }
    __syncthreads();
    const int before = warp ? buffer[warp - 1] : 0;
    const int out = MAX ? max(v, before) : v + before;
    __syncthreads();
    return out;
}

// One block of P threads (a power of two >= routes): the routes sorted by (expert, route) by a bitonic sort in
// shared memory, then cut into items of `width`.
__global__ void plan_kernel(const long long* ids64, const int* ids32, int n, int experts, const int* slots,
                            Items items) {
    extern __shared__ unsigned keys[];
    __shared__ int buffer[32];
    const int t = threadIdx.x, P = blockDim.x;
    unsigned key = 0xFFFFFFFFu;
    if(t < n) {
        const long long e = ids64 ? ids64[t] : static_cast<long long>(ids32[t]);
        int s = -1;
        if(e >= 0 && e < experts) s = slots ? slots[e] : static_cast<int>(e);
        key = s >= 0 ? (static_cast<unsigned>(e) << 10) | static_cast<unsigned>(t) : 0xFFFFFC00u | static_cast<unsigned>(t);
    }
    keys[t] = key;
    for(int i = t; i < items.capacity * items.width; i += P) items.routes[i] = -1;
    __syncthreads();
    for(int size = 2; size <= P; size <<= 1)
        for(int stride = size >> 1; stride > 0; stride >>= 1) {
            const int j = t ^ stride;
            if(j > t) {
                const unsigned a = keys[t], b = keys[j];
                if((a > b) == ((t & size) == 0)) {
                    keys[t] = b;
                    keys[j] = a;
                }
            }
            __syncthreads();
        }
    const unsigned k = keys[t];
    const bool valid = t < n && k < 0xFFFFFC00u;
    const bool first = valid && (t == 0 || (keys[t - 1] >> 10) != (k >> 10));
    const int start = block_scan<true>(first ? t : 0, buffer);
    const int rank = t - start;
    const int flag = valid && rank % items.width == 0;
    const int index = block_scan<false>(flag, buffer) - 1;
    if(valid) {
        items.routes[index * items.width + rank % items.width] = static_cast<int>(k & 1023u);
        if(flag) items.slot[index] = slots ? slots[k >> 10] : static_cast<int>(k >> 10);
    }
    if(t == P - 1) *items.count = index + 1;
}

__global__ void items_kernel(const long long* order, const int* counts, const int* starts, const int* item_end,
                             int experts, const int* slots, Items items) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int u = static_cast<int>(i / items.width), j = static_cast<int>(i % items.width);
    if(u >= items.capacity) return;
    const int total = item_end[experts - 1];
    if(i == 0) *items.count = total;
    if(u >= total) {
        items.routes[i] = -1;
        return;
    }
    int lo = 0, hi = experts - 1;   // the first expert whose items end past u
    while(lo < hi) {
        const int mid = (lo + hi) / 2;
        if(item_end[mid] > u) hi = mid;
        else lo = mid + 1;
    }
    const int e = lo, base = e ? item_end[e - 1] : 0;
    const int pos = (u - base) * items.width + j;
    items.routes[i] = pos < counts[e] ? static_cast<int>(order[starts[e] + pos]) : -1;
    if(j == 0) items.slot[u] = slots ? slots[e] : e;
}

__global__ void combine_kernel(const float* y, const long long* ids64, const int* ids32, const float* weights,
                               const int* slots, int experts, int top, int hidden, bool skip, float* out) {
    const int t = blockIdx.y;
    const int c = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(c >= hidden) return;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for(int j = 0; j < top; ++j) {
        const long long r = static_cast<long long>(t) * top + j;
        const long long e = ids64 ? ids64[r] : static_cast<long long>(ids32[r]);
        if(e < 0 || e >= experts || (skip && slots && slots[e] < 0)) continue;
        const float w = weights[r];
        const float4 v = *reinterpret_cast<const float4*>(y + r * hidden + c);
        acc.x = __fmaf_rn(w, v.x, acc.x);
        acc.y = __fmaf_rn(w, v.y, acc.y);
        acc.z = __fmaf_rn(w, v.z, acc.z);
        acc.w = __fmaf_rn(w, v.w, acc.w);
    }
    *reinterpret_cast<float4*>(out + static_cast<long long>(t) * hidden + c) = acc;
}

__global__ void quantize_kernel(const float* x, int rows, int cols, float g, std::int8_t* values, float* scales) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int blocks = cols / 16;
    if(i >= static_cast<long long>(rows) * blocks) return;
    const float* v = x + i * 16;
    float amax = 0.0f;
    for(int k = 0; k < 16; ++k) amax = fmaxf(amax, fabsf(v[k]));
    const float decoded = block_scale(amax, g);
    for(int k = 0; k < 16; ++k) {
        float value = 0.0f;
        if(decoded > 0.0f) {
            const float q = __fdiv_rn(__fmul_rn(v[k], g), decoded);
            const int code = e2m1_code(fabsf(q));
            value = 2.0f * e2m1_magnitude(code) * ((q < 0.0f && code) ? -1.0f : 1.0f);
        }
        values[i * 16 + k] = static_cast<std::int8_t>(value);
    }
    scales[i] = __fdiv_rn(decoded, g);
}

// ----------------------------------------------------------------------------------------------- launches

constexpr int kWarps = 4;

int chunk_for(int width, int K) {   // k values staged per pass: a decode item stages all of K when it fits 48 KiB
    if(width == 8) return 8 * (K + kPad) * 2 <= 48 * 1024 ? K : 1024;
    return 256;
}

template <int F, int NT, bool GATEUP, bool A4>
cudaError_t launch(const Pass& base, const Items& items, cudaStream_t stream) {
    constexpr int TILES = GATEUP ? 2 : (NT == 1 ? 1 : 2);
    Pass p = base;
    const int K = GATEUP ? p.layout.hidden : p.layout.inter, M = GATEUP ? p.layout.inter : p.layout.hidden;
    p.chunk = chunk_for(items.width, K);
    const int rows_per_block = kWarps * 16 * (GATEUP ? 1 : TILES);
    const dim3 grid((M + rows_per_block - 1) / rows_per_block, items.capacity);
    const size_t bytes = static_cast<size_t>(8 * NT) * (p.chunk + kPad) * sizeof(bf16);
    expert_kernel<F, NT, TILES, kWarps, GATEUP, A4><<<grid, kWarps * 32, bytes, stream>>>(p);
    return cudaGetLastError();
}

template <bool GATEUP>
cudaError_t dispatch(const Pass& p, bool a4, const Items& items, cudaStream_t stream) {
    const int f = static_cast<int>(p.layout.format);
    if(a4 && p.layout.format != Format::nvfp4) return cudaErrorInvalidValue;
    const bool wide = items.width == 32;
    if(items.width != 8 && !wide) return cudaErrorInvalidValue;
#define CRIT_GX_CASE(FMT)                                                                                          \
    case int(FMT):                                                                                                 \
        return wide ? launch<int(FMT), 4, GATEUP, false>(p, items, stream) : launch<int(FMT), 1, GATEUP, false>(p, items, stream);
    if(a4) return wide ? launch<int(Format::nvfp4), 4, GATEUP, true>(p, items, stream)
                       : launch<int(Format::nvfp4), 1, GATEUP, true>(p, items, stream);
    switch(f) {
        CRIT_GX_CASE(Format::bf16)
        CRIT_GX_CASE(Format::fp8_channel)
        CRIT_GX_CASE(Format::fp8_block)
        CRIT_GX_CASE(Format::nvfp4)
        CRIT_GX_CASE(Format::int4)
        CRIT_GX_CASE(Format::mxfp4)
        default:
            return cudaErrorInvalidValue;
    }
#undef CRIT_GX_CASE
}

bool valid(const Layout& L) {
    return L.bytes > 0 && L.bytes % 16 == 0 && L.hidden >= 64 && L.inter >= 64 && L.hidden % 64 == 0 &&
           L.inter % 64 == 0 && L.gate_up >= 0 && L.down >= 0;
}
bool valid(const Items& it) {
    return it.slot && it.routes && it.count && (it.width == 8 || it.width == 32) && it.capacity >= 0;
}

}  // namespace

cudaError_t enqueue_plan(const long long* ids64, const int* ids32, int routes, int experts, const int* slots,
                         const Items& items, cudaStream_t stream) {
    if((!ids64 && !ids32) || routes < 0 || routes > kSmallPlanRoutes || experts < 1 || experts >= (1 << 22) - 1 ||
       !valid(items))
        return cudaErrorInvalidValue;
    int threads = 32;
    while(threads < routes) threads <<= 1;
    plan_kernel<<<1, threads, threads * sizeof(unsigned), stream>>>(ids64, ids32, routes, experts, slots, items);
    return cudaGetLastError();
}

cudaError_t enqueue_items(const long long* order, const int* counts, const int* starts, const int* item_end,
                          int experts, const int* slots, const Items& items, cudaStream_t stream) {
    if(!order || !counts || !starts || !item_end || experts < 1 || !valid(items)) return cudaErrorInvalidValue;
    const long long n = static_cast<long long>(items.capacity) * items.width;
    if(n == 0) return cudaSuccess;
    items_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(order, counts, starts, item_end, experts,
                                                                              slots, items);
    return cudaGetLastError();
}

cudaError_t enqueue_gate_up(const bf16* x, int top, const std::uint8_t* store, const Layout& layout, bool a4,
                            const Items& items, bf16* h, cudaStream_t stream) {
    if(!x || top < 1 || !store || !h || !valid(layout) || !valid(items) || (a4 && layout.input < 0))
        return cudaErrorInvalidValue;
    if(items.capacity == 0) return cudaSuccess;
    const Pass p{store, layout, items.slot, items.routes, items.count, x, top, h, 0};
    return dispatch<true>(p, a4, items, stream);
}

cudaError_t enqueue_down(const bf16* h, const std::uint8_t* store, const Layout& layout, bool a4, const Items& items,
                         float* y, cudaStream_t stream) {
    if(!h || !store || !y || !valid(layout) || !valid(items) || (a4 && layout.input < 0)) return cudaErrorInvalidValue;
    if(items.capacity == 0) return cudaSuccess;
    const Pass p{store, layout, items.slot, items.routes, items.count, h, 1, y, 0};
    return dispatch<false>(p, a4, items, stream);
}

cudaError_t enqueue_combine(const float* y, const long long* ids64, const int* ids32, const float* weights,
                            const int* slots, int experts, int tokens, int top, int hidden, bool skip_negative,
                            float* out, cudaStream_t stream) {
    if(!y || (!ids64 && !ids32) || !weights || !out || experts < 1 || tokens < 0 || top < 1 || hidden % 4)
        return cudaErrorInvalidValue;
    if(tokens == 0) return cudaSuccess;
    const dim3 grid((hidden / 4 + 127) / 128, tokens);
    combine_kernel<<<grid, 128, 0, stream>>>(y, ids64, ids32, weights, slots, experts, top, hidden, skip_negative, out);
    return cudaGetLastError();
}

cudaError_t enqueue_quantize_e2m1x2(const float* x, int rows, int cols, float global, std::int8_t* values,
                                    float* scales, cudaStream_t stream) {
    if(!x || !values || !scales || rows < 0 || cols % 16 || !(global > 0.0f)) return cudaErrorInvalidValue;
    const long long n = static_cast<long long>(rows) * (cols / 16);
    if(n == 0) return cudaSuccess;
    quantize_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(x, rows, cols, global, values, scales);
    return cudaGetLastError();
}

}  // namespace crit::techniques::grouped_experts
