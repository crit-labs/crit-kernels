// Matched FP8 projections (fp8_block.h). The arithmetic is crit's fp8_block_device.cuh, operation for operation: one
// e4m3 m16n8k32 MMA with a zero accumulator per 32-element block, one rounded scale product and one FMA per block on
// lane b % 16, the halving tree, post. Explicit round-to-nearest intrinsics keep the compiler from contracting anything.
#include "fp8_block.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace crit::techniques::fp8_block {
namespace {

constexpr int kBlock = 32, kLanes = 16;

// Bytes 8q .. 8q + 7 of block b of one row or token (crit's fragment order).
__device__ inline uint2 fragment(const std::uint8_t* row, int b, int q) {
    return *reinterpret_cast<const uint2*>(row + kBlock * b + 8 * q);
}

// d[2h + c]: row g + 8h against the token in column 2q + c.
__device__ inline void mma_block(uint2 a_g, uint2 a_g8, uint2 t_g, float (&d)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %10, %10, %10};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a_g.x), "r"(a_g8.x), "r"(a_g.y), "r"(a_g8.y), "r"(t_g.x), "r"(t_g.y), "f"(0.0f));
#else
    (void)a_g, (void)a_g8, (void)t_g;
    d[0] = d[1] = d[2] = d[3] = 0.0f;
    __trap();
#endif
}

__device__ inline float step(float sum, float ws, float as, float lane) { return __fmaf_rn(sum, __fmul_rn(ws, as), lane); }

// A weight scale k of a matrix's scale row: FP32, or an E8M0 byte (2^(e - 127); 0 is 2^-127, 255 NaN).
__device__ inline float weight_scale(const std::uint8_t* row, int k, int e8m0) {
    if(!e8m0) return reinterpret_cast<const float*>(row)[k];
    const unsigned e = row[k];
    return __uint_as_float(e == 0u ? 0x00400000u : e == 255u ? 0x7fc00000u : e << 23);
}

// The 16 lanes in the order the halving tree consumes them; fold adds lane kLaneOrder[at]'s finished sum into the
// partial sums x[0..3] in the tree's pairs.
__device__ constexpr int kLaneOrder[kLanes] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
__device__ inline void fold(int at, float a, float* x) {
    switch(at) {
        case 0: x[0] = a; break;
        case 1: x[0] = __fadd_rn(x[0], a); break;
        case 2: x[1] = a; break;
        case 3: x[1] = __fadd_rn(x[1], a); x[0] = __fadd_rn(x[0], x[1]); break;
        case 4: x[1] = a; break;
        case 5: x[1] = __fadd_rn(x[1], a); break;
        case 6: x[2] = a; break;
        case 7: x[2] = __fadd_rn(x[2], a); x[1] = __fadd_rn(x[1], x[2]); x[0] = __fadd_rn(x[0], x[1]); break;
        case 8: x[1] = a; break;
        case 9: x[1] = __fadd_rn(x[1], a); break;
        case 10: x[2] = a; break;
        case 11: x[2] = __fadd_rn(x[2], a); x[1] = __fadd_rn(x[1], x[2]); break;
        case 12: x[2] = a; break;
        case 13: x[2] = __fadd_rn(x[2], a); break;
        case 14: x[3] = a; break;
        default: x[3] = __fadd_rn(x[3], a); x[2] = __fadd_rn(x[2], x[3]); x[1] = __fadd_rn(x[1], x[2]);
                 x[0] = __fadd_rn(x[0], x[1]); break;
    }
}

// ---------------------------------------------------------------------------------------------------- decode rows

// A CTA computes 16 rows of one job: warp w computes lane w (kRowWarps 16) with kAhead rounds of loads in flight, then
// one thread an output reduces its 16 lanes by the halving tree in shared memory. A missing row or token reads a
// present one's bytes and is never written: the tensor core's outputs for other rows and tokens do not depend on it.
// On an RTX 5090 this reads a 13 MB matrix at the rate of a plain streaming read of it.
constexpr int kRowWarps = 16, kRowThreads = 32 * kRowWarps, kSlots = kLanes / kRowWarps, kAhead = 2;

__global__ void __launch_bounds__(kRowThreads) rows_kernel(const Matrix* matrices, const RowJob* jobs, Acts acts,
                                                           float* out, int out_ld) {
    __shared__ float part[kLanes][16][kMaxJobTokens + 1];
    __shared__ int act_row[kMaxJobTokens], out_row[kMaxJobTokens];
    const RowJob* job = jobs + blockIdx.y;
    const Matrix m = matrices[job->matrix];
    const int row0 = int(blockIdx.x) * 16;
    if(row0 >= m.rows) return;   // the whole CTA
    const int tokens = job->tokens;
    if(threadIdx.x < kMaxJobTokens) {
        act_row[threadIdx.x] = int(threadIdx.x) < tokens ? job->act_row[threadIdx.x] : job->act_row[0];
        out_row[threadIdx.x] = job->out_row[threadIdx.x];
    }
    __syncthreads();
    const int warp = int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32, g = lane / 4, q = lane % 4;
    const int ra = min(row0 + g, m.rows - 1), rb = min(row0 + g + 8, m.rows - 1);
    const std::uint8_t* wa = m.codes + std::size_t(ra) * std::size_t(m.cols);
    const std::uint8_t* wb = m.codes + std::size_t(rb) * std::size_t(m.cols);
    const std::uint8_t* xg = acts.codes + std::size_t(act_row[g]) * std::size_t(acts.ld);
    const std::size_t sld = std::size_t(m.cols / m.scale_k) * (m.scale_e8m0 ? 1u : 4u);
    const std::uint8_t* wsa = static_cast<const std::uint8_t*>(m.scales) + std::size_t(ra / m.scale_rows) * sld;
    const std::uint8_t* wsb = static_cast<const std::uint8_t*>(m.scales) + std::size_t(rb / m.scale_rows) * sld;
    const float* xs0 = acts.scales + std::size_t(act_row[2 * q]) * std::size_t(acts.lds);
    const float* xs1 = acts.scales + std::size_t(act_row[2 * q + 1]) * std::size_t(acts.lds);
    const int blocks = m.cols / kBlock;
    float acc[kSlots][4];
#pragma unroll
    for(int s = 0; s < kSlots; ++s)
#pragma unroll
        for(int o = 0; o < 4; ++o) acc[s][o] = 0.0f;
    // kAhead rounds of loads issued before their MMAs; each lane's blocks still in increasing order.
    for(int base = 0; base < blocks; base += kLanes * kAhead) {
        uint2 fa[kAhead][kSlots], fb[kAhead][kSlots], fx[kAhead][kSlots];
        float w0[kAhead][kSlots], w1[kAhead][kSlots], a0[kAhead][kSlots], a1[kAhead][kSlots];
#pragma unroll
        for(int u = 0; u < kAhead; ++u)
#pragma unroll
            for(int s = 0; s < kSlots; ++s) {
                const int b = min(base + kLanes * u + warp + kRowWarps * s, blocks - 1);
                fa[u][s] = fragment(wa, b, q);
                fb[u][s] = fragment(wb, b, q);
                fx[u][s] = fragment(xg, b, q);
                const int wk = b * kBlock / m.scale_k, xk = b * kBlock / acts.group;
                w0[u][s] = weight_scale(wsa, wk, m.scale_e8m0);
                w1[u][s] = weight_scale(wsb, wk, m.scale_e8m0);
                a0[u][s] = xs0[xk];
                a1[u][s] = xs1[xk];
            }
#pragma unroll
        for(int u = 0; u < kAhead; ++u)
#pragma unroll
            for(int s = 0; s < kSlots; ++s) {
                if(base + kLanes * u + warp + kRowWarps * s >= blocks) continue;
                float d[4];
                mma_block(fa[u][s], fb[u][s], fx[u][s], d);
                acc[s][0] = step(d[0], w0[u][s], a0[u][s], acc[s][0]);
                acc[s][1] = step(d[1], w0[u][s], a1[u][s], acc[s][1]);
                acc[s][2] = step(d[2], w1[u][s], a0[u][s], acc[s][2]);
                acc[s][3] = step(d[3], w1[u][s], a1[u][s], acc[s][3]);
            }
    }
#pragma unroll
    for(int s = 0; s < kSlots; ++s)
#pragma unroll
        for(int o = 0; o < 4; ++o) part[warp + kRowWarps * s][g + 8 * (o / 2)][2 * q + o % 2] = acc[s][o];
    __syncthreads();
    if(int(threadIdx.x) >= 16 * kMaxJobTokens) return;
    const int pr = int(threadIdx.x) / kMaxJobTokens, pt = int(threadIdx.x) % kMaxJobTokens;
    if(row0 + pr >= m.rows || pt >= tokens) return;
    float v[kLanes];
#pragma unroll
    for(int j = 0; j < kLanes; ++j) v[j] = part[j][pr][pt];
#pragma unroll
    for(int w = kLanes / 2; w >= 1; w /= 2)
#pragma unroll
        for(int i = 0; i < w; ++i) v[i] = __fadd_rn(v[i], v[i + w]);
    out[std::size_t(out_row[pt]) * std::size_t(out_ld) + std::size_t(row0 + pr)] = __fmul_rn(v[0], m.post);
}

// ---------------------------------------------------------------------------------------------------- prompt GEMM

// A warp computes 16 rows x 32 tokens (four n-tiles sharing the A fragment) per pass, lane by lane in kLaneOrder: lane
// j's blocks j, j + 16, ... in order, each block's sum from the tensor cores and the FP32 step on the CUDA cores, then
// the lane folded into the tree. Token tiles of 32 spread over grid.z.
constexpr int kGemmWarps = 8, kTiles = 4, kGemmRows = 16 * kGemmWarps, kGemmTokens = 8 * kTiles;

__global__ void __launch_bounds__(32 * kGemmWarps) gemm_kernel(const Matrix* matrices, const GemmJob* jobs, Acts acts,
                                                               float* out, int out_ld) {
    const GemmJob job = jobs[blockIdx.y];
    const Matrix m = matrices[job.matrix];
    const int warp = int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32, g = lane / 4, q = lane % 4;
    const int row0 = int(blockIdx.x) * kGemmRows + warp * 16;
    if(row0 >= m.rows || job.count <= 0) return;
    const bool live_a = row0 + g < m.rows, live_b = row0 + g + 8 < m.rows;
    const int ra = live_a ? row0 + g : m.rows - 1, rb = live_b ? row0 + g + 8 : m.rows - 1;
    const std::uint8_t* wa = m.codes + std::size_t(ra) * std::size_t(m.cols);
    const std::uint8_t* wb = m.codes + std::size_t(rb) * std::size_t(m.cols);
    const std::size_t sld = std::size_t(m.cols / m.scale_k) * (m.scale_e8m0 ? 1u : 4u);
    const std::uint8_t* wsa = static_cast<const std::uint8_t*>(m.scales) + std::size_t(ra / m.scale_rows) * sld;
    const std::uint8_t* wsb = static_cast<const std::uint8_t*>(m.scales) + std::size_t(rb / m.scale_rows) * sld;
    const int blocks = m.cols / kBlock, last = job.first + job.count - 1;
    for(int tile = kGemmTokens * int(blockIdx.z); tile < job.count; tile += kGemmTokens * int(gridDim.z)) {
        const std::uint8_t* xg[kTiles];
        const float* xs[kTiles][2];
#pragma unroll
        for(int n = 0; n < kTiles; ++n) {
            const int t = min(job.first + tile + 8 * n + g, last);
            xg[n] = acts.codes + std::size_t(t) * std::size_t(acts.ld);
#pragma unroll
            for(int c = 0; c < 2; ++c)
                xs[n][c] = acts.scales + std::size_t(min(job.first + tile + 8 * n + 2 * q + c, last)) * std::size_t(acts.lds);
        }
        float x[4 * kTiles][4];
#pragma unroll
        for(int at = 0; at < kLanes; ++at) {
            const int j = kLaneOrder[at];
            float acc[4 * kTiles];
#pragma unroll
            for(int o = 0; o < 4 * kTiles; ++o) acc[o] = 0.0f;
            for(int b = j; b < blocks; b += kLanes) {
                const uint2 a_g = fragment(wa, b, q), a_g8 = fragment(wb, b, q);
                const int wk = b * kBlock / m.scale_k, xk = b * kBlock / acts.group;
                const float w0 = weight_scale(wsa, wk, m.scale_e8m0), w1 = weight_scale(wsb, wk, m.scale_e8m0);
#pragma unroll
                for(int n = 0; n < kTiles; ++n) {
                    float d[4];
                    mma_block(a_g, a_g8, fragment(xg[n], b, q), d);
                    const float a0 = xs[n][0][xk], a1 = xs[n][1][xk];
                    acc[4 * n + 0] = step(d[0], w0, a0, acc[4 * n + 0]);
                    acc[4 * n + 1] = step(d[1], w0, a1, acc[4 * n + 1]);
                    acc[4 * n + 2] = step(d[2], w1, a0, acc[4 * n + 2]);
                    acc[4 * n + 3] = step(d[3], w1, a1, acc[4 * n + 3]);
                }
            }
#pragma unroll
            for(int o = 0; o < 4 * kTiles; ++o) fold(at, acc[o], x[o]);
        }
#pragma unroll
        for(int n = 0; n < kTiles; ++n)
#pragma unroll
            for(int o = 0; o < 4; ++o) {
                const int t = tile + 8 * n + 2 * q + o % 2;
                if(t < job.count && (o / 2 ? live_b : live_a))
                    out[std::size_t(job.first + t) * std::size_t(out_ld) + std::size_t(o / 2 ? row0 + g + 8 : row0 + g)] =
                        __fmul_rn(x[4 * n + o][0], m.post);
            }
    }
}

// ---------------------------------------------------------------------------------------------------- quantizer

// crit's quantize_group_device: MXFP8's 2^(floor(log2 amax) - 8) clamped to E8M0's range, or amax / 448; E4M3 by the
// hardware conversion (round to nearest even, saturating). An all-zero group: scale 1 (mx) or 0 (amax), zero codes.
__global__ void quantize_kernel(const void* x, int bf16, int ld_x, int rows, int cols, ScaleKind kind, Acts acts) {
    const int groups = cols / acts.group;
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if(i >= rows * groups) return;
    const int row = i / groups, k0 = (i % groups) * acts.group;
    auto value = [&](int k) {
        const std::size_t at = std::size_t(row) * std::size_t(ld_x) + std::size_t(k0 + k);
        return bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(x)[at]) : static_cast<const float*>(x)[at];
    };
    float amax = 0.0f;
    for(int k = 0; k < acts.group; ++k) amax = fmaxf(amax, fabsf(value(k)));
    float scale;
    if(kind == ScaleKind::mx) {
        int e = 0;
        (void)frexpf(amax, &e);
        int shared = amax > 0.0f ? e - 1 - 8 : 0;
        shared = shared < -127 ? -127 : shared > 127 ? 127 : shared;
        scale = ldexpf(1.0f, shared);
    } else {
        scale = __fdiv_rn(amax, 448.0f);
    }
    std::uint8_t* codes = const_cast<std::uint8_t*>(acts.codes) + std::size_t(row) * std::size_t(acts.ld) + std::size_t(k0);
    for(int k = 0; k < acts.group; ++k)
        codes[k] = scale > 0.0f
                       ? std::uint8_t(__nv_cvt_float_to_fp8(__fdiv_rn(value(k), scale), __NV_SATFINITE, __NV_E4M3))
                       : std::uint8_t(0);
    const_cast<float*>(acts.scales)[std::size_t(row) * std::size_t(acts.lds) + std::size_t(i % groups)] = scale;
}

bool bad_acts(const Acts& a) { return !a.codes || !a.scales || a.group <= 0 || a.group % kBlock || a.ld % 8; }

}  // namespace

cudaError_t enqueue_rows(const Matrix* matrices, const RowJob* jobs, int job_count, int max_rows, const Acts& acts,
                         float* out, int out_ld, cudaStream_t stream) {
    if(!matrices || !jobs || job_count < 0 || max_rows < 0 || !out || bad_acts(acts)) return cudaErrorInvalidValue;
    if(job_count == 0 || max_rows == 0) return cudaSuccess;
    const dim3 grid(unsigned((max_rows + 15) / 16), unsigned(job_count));
    rows_kernel<<<grid, kRowThreads, 0, stream>>>(matrices, jobs, acts, out, out_ld);
    return cudaGetLastError();
}

cudaError_t enqueue_gemm(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows, int max_count,
                         const Acts& acts, float* out, int out_ld, cudaStream_t stream) {
    if(!matrices || !jobs || job_count < 0 || max_rows < 0 || max_count < 0 || !out || bad_acts(acts))
        return cudaErrorInvalidValue;
    if(job_count == 0 || max_rows == 0 || max_count == 0) return cudaSuccess;
    const int tiles = (max_count + kGemmTokens - 1) / kGemmTokens;
    const dim3 grid(unsigned((max_rows + kGemmRows - 1) / kGemmRows), unsigned(job_count),
                    unsigned(tiles < 65535 ? tiles : 65535));
    gemm_kernel<<<grid, 32 * kGemmWarps, 0, stream>>>(matrices, jobs, acts, out, out_ld);
    return cudaGetLastError();
}

cudaError_t enqueue_quantize(const void* x, bool bf16, int ld_x, int rows, int cols, ScaleKind kind, const Acts& acts,
                             cudaStream_t stream) {
    if(!x || rows < 0 || cols < 0 || bad_acts(acts) || cols % acts.group) return cudaErrorInvalidValue;
    const int n = rows * (cols / acts.group);
    if(n == 0) return cudaSuccess;
    quantize_kernel<<<(n + 127) / 128, 128, 0, stream>>>(x, bf16 ? 1 : 0, ld_x, rows, cols, kind, acts);
    return cudaGetLastError();
}

}  // namespace crit::techniques::fp8_block
