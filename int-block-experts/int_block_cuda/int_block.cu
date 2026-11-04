// Integer block expert projections (int_block.h). The device arithmetic is crit's host_int_device.cuh, operation for
// operation: exact integer block sums, then one rounded product and one FMA per block on lane b % 16, the halving
// tree, post. Explicit round-to-nearest intrinsics keep the compiler from contracting anything.
#include "int_block.h"

#include <cstdint>

namespace crit::techniques::int_block {
namespace {

// Integer values by code, in registers: E2M1 x 2 magnitudes {0, 1, 2, 3, 4, 6, 8, 12} as two packed words for
// __byte_perm; INT4 by sign extension.
__device__ inline int weight_value(unsigned code, int int4) {
    if(int4) return int(code << 28) >> 28;
    const int magnitude = int(__byte_perm(0x03020100u, 0x0c080604u, code & 7u) & 0xffu);
    return (code & 8u) ? -magnitude : magnitude;
}
// Four consecutive weight codes (element k, k % 4 == 0) as packed int8 values.
__device__ inline int four_weights_of(unsigned pair, int int4) {
    unsigned packed = 0;
#pragma unroll
    for(int i = 0; i < 4; ++i) packed |= (unsigned(weight_value((pair >> (4 * i)) & 15u, int4)) & 0xffu) << (8 * i);
    return int(packed);
}
__device__ inline int four_weights(const std::uint8_t* codes, int k, int int4) {
    return four_weights_of(*reinterpret_cast<const unsigned short*>(codes + k / 2), int4);
}
// E4M3 scale values, decoded once per launch into shared memory by e4m3_table.
__device__ inline float e4m3(unsigned byte) {
    const int exponent = int(byte >> 3) & 15, mantissa = int(byte) & 7;
    const float magnitude = exponent == 0 ? ldexpf(float(mantissa) / 8.0f, -6)
                                          : ldexpf(1.0f + float(mantissa) / 8.0f, exponent - 7);
    return (byte & 0x80u) ? -magnitude : magnitude;
}
__device__ inline float e2m1(unsigned code) {
    const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (code & 8u) ? -magnitude[code & 7u] : magnitude[code & 7u];
}
__device__ inline unsigned e2m1_encode(float x) {
    const float m = fminf(fabsf(x), 6.0f);
    const float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    unsigned code = 0;
    while(code < 7 && (m > mids[code] || (m == mids[code] && (code % 2) == 1))) ++code;
    return code | (x < 0.0f && code ? 8u : 0u);
}
__device__ inline unsigned e4m3_encode(float x) {
    const float m = fminf(fabsf(x), 448.0f);
    unsigned best = 0;
    for(unsigned i = 1; i < 0x7Fu; ++i) {
        const float d = fabsf(m - e4m3(i)), bd = fabsf(m - e4m3(best));
        if(d < bd || (d == bd && i % 2 == 0 && best % 2 == 1)) best = i;
    }
    return best | (x < 0.0f && best ? 0x80u : 0u);
}
__device__ inline float exp_declared(float x) {
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

// ---------------------------------------------------------------------------------------------------- decode rows

constexpr int kRowThreads = 256, kRowsPerCta = kRowThreads / 16;

__device__ inline void e4m3_table(float* table) {
    for(int i = int(threadIdx.x); i < 256; i += int(blockDim.x)) table[i] = e4m3(unsigned(i));
    __syncthreads();
}

__global__ void rows_kernel(const Matrix* matrices, const RowJob* jobs, Acts acts, float* out, int out_ld) {
    __shared__ float scale_of[256];
    e4m3_table(scale_of);
    const RowJob& job = jobs[blockIdx.y];
    const Matrix m = matrices[job.matrix];
    const int row = int(blockIdx.x) * kRowsPerCta + int(threadIdx.x) / 16;
    if(row >= m.rows) return;   // the whole half-warp leaves together: the shuffles stay within it
    const unsigned lane = threadIdx.x % 16u;
    const unsigned group = 0xffffu << (threadIdx.x % 32u & 16u);
    const int blocks = m.cols / 16, ratio = acts.block / 16, tokens = job.tokens;
    const std::uint8_t* codes = m.codes + std::size_t(row) * std::size_t(m.cols / 2);
    const std::uint8_t* scales = m.scales + std::size_t(row) * std::size_t(blocks);
    float acc[kMaxJobTokens];
#pragma unroll
    for(int t = 0; t < kMaxJobTokens; ++t) acc[t] = 0.0f;
    for(int b = int(lane); b < blocks; b += 16) {
        const uint2 q = *reinterpret_cast<const uint2*>(codes + 8 * b);   // the block's 16 codes
        const int w0 = four_weights_of(q.x & 0xffffu, m.int4), w1 = four_weights_of(q.x >> 16, m.int4);
        const int w2 = four_weights_of(q.y & 0xffffu, m.int4), w3 = four_weights_of(q.y >> 16, m.int4);
        const float ws = scale_of[scales[b]];
#pragma unroll
        for(int t = 0; t < kMaxJobTokens; ++t) {
            if(t >= tokens) break;
            const int4 x = *reinterpret_cast<const int4*>(acts.values + std::size_t(job.act_row[t]) * acts.ld + 16 * b);
            int sum = __dp4a(w0, x.x, 0);
            sum = __dp4a(w1, x.y, sum);
            sum = __dp4a(w2, x.z, sum);
            sum = __dp4a(w3, x.w, sum);
            const float as = acts.scales[std::size_t(job.act_row[t]) * acts.lds + b / ratio];
            acc[t] = __fmaf_rn(float(sum), __fmul_rn(ws, as), acc[t]);
        }
    }
#pragma unroll
    for(int t = 0; t < kMaxJobTokens; ++t) {
        if(t >= tokens) break;
        float v = acc[t];
        for(int w = 8; w >= 1; w /= 2) v = __fadd_rn(v, __shfl_down_sync(group, v, w, 16));
        if(lane == 0) out[std::size_t(job.out_row[t]) * out_ld + row] = __fmul_rn(v, m.post);
    }
}

// ---------------------------------------------------------------------------------------------------- prompt GEMM

constexpr int kGemmWarps = 8, kTiles = 4, kGemmRows = 8 * kTiles * kGemmWarps;   // a warp: 16 tokens x 32 rows

__device__ inline void mma_s8(int a0, int a1, int b, int& c0, int& c1, int& c2, int& c3) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, "
                 "{%7, %7, %7, %7};\n"
                 : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                 : "r"(a0), "r"(a1), "r"(b), "r"(0));
}

// The 16 lanes in the order the halving tree consumes them, so a lane's sum joins the tree as soon as it is done and
// an output holds one running lane and at most four partial sums instead of 16 lanes:
// ((a0 + a8) + (a4 + a12)) + ((a2 + a10) + (a6 + a14)), then the same over the odd lanes, then their sum.
__device__ constexpr int kLaneOrder[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

// Folds lane `step`'s finished sum `a` (in kLaneOrder) into the partial sums x[0..3]: the halving tree's adds, in its
// pairs (IEEE addition is commutative, so only the tree's shape matters).
__device__ inline void fold(int step, float a, float* x) {
    switch(step) {
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

// A warp computes 16 tokens x 32 weight rows (four m16n8k16 tiles sharing the A fragment) per pass over the blocks,
// lane by lane in kLaneOrder: lane j's blocks j, j + 16, ... in order, each block's exact int8 sums from the tensor
// cores and the FP32 step on the CUDA cores, then the lane folded into the tree.
__global__ void gemm_kernel(const Matrix* matrices, const GemmJob* jobs, Acts acts, float* out, int out_ld) {
    __shared__ float scale_of[256];
    e4m3_table(scale_of);
    const GemmJob job = jobs[blockIdx.y];
    const Matrix m = matrices[job.matrix];
    const int warp = int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32;
    const int g = lane / 4, tig = lane % 4;
    const int base = int(blockIdx.x) * kGemmRows + warp * 8 * kTiles;
    if(base >= m.rows) return;
    const int blocks = m.cols / 16, ratio = acts.block / 16;
    const std::uint8_t* wcodes[kTiles];
    const std::uint8_t* wscale[kTiles][2];
    bool wlive[kTiles], olive[kTiles][2];
#pragma unroll
    for(int n = 0; n < kTiles; ++n) {
        const int wrow = base + 8 * n + g;   // the B fragment's row
        wlive[n] = wrow < m.rows;
        wcodes[n] = m.codes + std::size_t(wlive[n] ? wrow : 0) * std::size_t(m.cols / 2);
#pragma unroll
        for(int c = 0; c < 2; ++c) {
            const int orow = base + 8 * n + 2 * tig + c;   // this thread's output rows
            olive[n][c] = orow < m.rows;
            wscale[n][c] = m.scales + std::size_t(olive[n][c] ? orow : 0) * std::size_t(blocks);
        }
    }
    for(int tile_t = 16 * int(blockIdx.z); tile_t < job.count; tile_t += 16 * int(gridDim.z)) {
        const int t0 = tile_t + g, t1 = t0 + 8;
        const bool live0 = t0 < job.count, live1 = t1 < job.count;
        const std::int8_t* x0 = acts.values + std::size_t(job.first + (live0 ? t0 : 0)) * acts.ld;
        const std::int8_t* x1 = acts.values + std::size_t(job.first + (live1 ? t1 : 0)) * acts.ld;
        const float* xs0 = acts.scales + std::size_t(job.first + (live0 ? t0 : 0)) * acts.lds;
        const float* xs1 = acts.scales + std::size_t(job.first + (live1 ? t1 : 0)) * acts.lds;
        // Outputs o = (n, token half h, column c) at o = 4n + 2h + c; x[o][..]: the tree's partial sums.
        float x[4 * kTiles][4];
#pragma unroll
        for(int step = 0; step < 16; ++step) {
            const int j = kLaneOrder[step];
            float acc[4 * kTiles];
#pragma unroll
            for(int o = 0; o < 4 * kTiles; ++o) acc[o] = 0.0f;
            for(int b = j; b < blocks; b += 16) {
                const int a0 = live0 ? *reinterpret_cast<const int*>(x0 + 16 * b + 4 * tig) : 0;
                const int a1 = live1 ? *reinterpret_cast<const int*>(x1 + 16 * b + 4 * tig) : 0;
                const float as0 = xs0[b / ratio], as1 = xs1[b / ratio];
#pragma unroll
                for(int n = 0; n < kTiles; ++n) {
                    const int bw = wlive[n] ? four_weights(wcodes[n], 16 * b + 4 * tig, m.int4) : 0;
                    int c0, c1, c2, c3;
                    mma_s8(a0, a1, bw, c0, c1, c2, c3);
                    const float w0 = scale_of[wscale[n][0][b]], w1 = scale_of[wscale[n][1][b]];
                    acc[4 * n + 0] = __fmaf_rn(float(c0), __fmul_rn(w0, as0), acc[4 * n + 0]);
                    acc[4 * n + 1] = __fmaf_rn(float(c1), __fmul_rn(w1, as0), acc[4 * n + 1]);
                    acc[4 * n + 2] = __fmaf_rn(float(c2), __fmul_rn(w0, as1), acc[4 * n + 2]);
                    acc[4 * n + 3] = __fmaf_rn(float(c3), __fmul_rn(w1, as1), acc[4 * n + 3]);
                }
            }
#pragma unroll
            for(int o = 0; o < 4 * kTiles; ++o) fold(step, acc[o], x[o]);
        }
#pragma unroll
        for(int n = 0; n < kTiles; ++n)
#pragma unroll
            for(int h = 0; h < 2; ++h)
#pragma unroll
                for(int c = 0; c < 2; ++c) {
                    const int t = h ? t1 : t0;
                    if((h ? live1 : live0) && olive[n][c])
                        out[std::size_t(job.first + t) * out_ld + base + 8 * n + 2 * tig + c] =
                            __fmul_rn(x[4 * n + 2 * h + c][0], m.post);
                }
    }
}

// ---------------------------------------------------------------------------------------------------- quantizers

__global__ void quantize_e2m1x2_kernel(const float* x, int ld_x, int rows, int cols, float global, Acts acts) {
    const int blocks = cols / 16;
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if(i >= rows * blocks) return;
    const int row = i / blocks, b = i % blocks;
    const float* v = x + std::size_t(row) * ld_x + 16 * b;
    float amax = 0.0f;
    for(int k = 0; k < 16; ++k) amax = fmaxf(amax, fabsf(v[k]));
    const float wanted = __fdiv_rn(__fmul_rn(amax, global), 6.0f);
    const float decoded = e4m3(e4m3_encode(isfinite(wanted) ? wanted : 448.0f));
    std::int8_t* out = const_cast<std::int8_t*>(acts.values) + std::size_t(row) * acts.ld + 16 * b;
    for(int k = 0; k < 16; ++k) {
        const float q = __fdiv_rn(__fmul_rn(v[k], global), decoded);
        out[k] = decoded > 0.0f && q == q ? std::int8_t(2.0f * e2m1(e2m1_encode(q))) : 0;
    }
    const_cast<float*>(acts.scales)[std::size_t(row) * acts.lds + b] = __fdiv_rn(decoded, global);
}

__global__ void quantize_int8_kernel(const float* x, int ld_x, int rows, int cols, Acts acts) {
    const int blocks = cols / acts.block;
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if(i >= rows * blocks) return;
    const int row = i / blocks, b = i % blocks;
    const float* v = x + std::size_t(row) * ld_x + acts.block * b;
    float amax = 0.0f;
    for(int k = 0; k < acts.block; ++k) amax = fmaxf(amax, fabsf(v[k]));
    const float d = __fdiv_rn(amax, 127.0f);
    std::int8_t* out = const_cast<std::int8_t*>(acts.values) + std::size_t(row) * acts.ld + acts.block * b;
    for(int k = 0; k < acts.block; ++k) {
        float q = d > 0.0f ? rintf(__fdiv_rn(v[k], d)) : 0.0f;
        q = q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q;
        out[k] = std::int8_t(q);
    }
    const_cast<float*>(acts.scales)[std::size_t(row) * acts.lds + b] = d;
}

__global__ void swiglu_kernel(const float* gate_up, int inter, int rows, float* out) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if(i >= rows * inter) return;
    const int row = i / inter, k = i % inter;
    const float gate = gate_up[std::size_t(row) * 2 * inter + k], up = gate_up[std::size_t(row) * 2 * inter + inter + k];
    const float silu = __fdiv_rn(gate, __fadd_rn(1.0f, exp_declared(-gate)));
    out[std::size_t(row) * inter + k] = __fmul_rn(silu, up);
}

bool bad_acts(const Acts& a) { return !a.values || !a.scales || (a.block != 16 && a.block != 32) || a.ld % 4; }

}  // namespace

cudaError_t enqueue_rows(const Matrix* matrices, const RowJob* jobs, int job_count, int max_rows, const Acts& acts,
                         float* out, int out_ld, cudaStream_t stream) {
    if(!matrices || !jobs || job_count < 0 || max_rows < 0 || !out || bad_acts(acts)) return cudaErrorInvalidValue;
    if(job_count == 0 || max_rows == 0) return cudaSuccess;
    const dim3 grid(unsigned((max_rows + kRowsPerCta - 1) / kRowsPerCta), unsigned(job_count));
    rows_kernel<<<grid, kRowThreads, 0, stream>>>(matrices, jobs, acts, out, out_ld);
    return cudaGetLastError();
}

cudaError_t enqueue_gemm(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows, int max_count,
                         const Acts& acts, float* out, int out_ld, cudaStream_t stream) {
    if(!matrices || !jobs || job_count < 0 || max_rows < 0 || max_count < 0 || !out || bad_acts(acts))
        return cudaErrorInvalidValue;
    if(job_count == 0 || max_rows == 0 || max_count == 0) return cudaSuccess;
    // Token tiles of 16 across grid.z, so a long chunk spreads over every SM.
    const int tiles = (max_count + 15) / 16;
    const dim3 grid(unsigned((max_rows + kGemmRows - 1) / kGemmRows), unsigned(job_count),
                    unsigned(tiles < 65535 ? tiles : 65535));
    gemm_kernel<<<grid, 32 * kGemmWarps, 0, stream>>>(matrices, jobs, acts, out, out_ld);
    return cudaGetLastError();
}

cudaError_t enqueue_quantize_e2m1x2(const float* x, int ld_x, int rows, int cols, float global, const Acts& acts,
                                    cudaStream_t stream) {
    if(!x || rows < 0 || cols % 16 || !(global > 0.0f) || bad_acts(acts)) return cudaErrorInvalidValue;
    const int n = rows * (cols / 16);
    if(n == 0) return cudaSuccess;
    quantize_e2m1x2_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, ld_x, rows, cols, global, acts);
    return cudaGetLastError();
}

cudaError_t enqueue_quantize_int8(const float* x, int ld_x, int rows, int cols, const Acts& acts, cudaStream_t stream) {
    if(!x || rows < 0 || bad_acts(acts) || cols % acts.block) return cudaErrorInvalidValue;
    const int n = rows * (cols / acts.block);
    if(n == 0) return cudaSuccess;
    quantize_int8_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, ld_x, rows, cols, acts);
    return cudaGetLastError();
}

cudaError_t enqueue_swiglu(const float* gate_up, int inter, int rows, float* out, cudaStream_t stream) {
    if(!gate_up || !out || inter < 0 || rows < 0) return cudaErrorInvalidValue;
    const int n = rows * inter;
    if(n == 0) return cudaSuccess;
    swiglu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(gate_up, inter, rows, out);
    return cudaGetLastError();
}

}  // namespace crit::techniques::int_block
