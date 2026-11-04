#include "grouped_experts.h"

#include <cuda_bf16.h>
#include <mma.h>

namespace crit::techniques::streamed_prefill {
namespace {

constexpr int kSortThreads = 256;
constexpr int kProjectionThreads = 128;  // four warps, each a 32 x 32 quarter of the 64 x 64 tile
constexpr int kPad = 8;                  // FP16 elements of padding per shared row (ldm stays a multiple of 8)

__device__ inline float e4m3(std::uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? ldexpf(float(m), -9) : ldexpf(float(8 + m), e - 10);
    if(e == 15 && m == 7) v = __int_as_float(0x7fc00000);
    return b & 0x80 ? -v : v;
}

__device__ inline float e2m1(unsigned code) {
    const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float m = magnitude[code & 7u];
    return code & 8u ? -m : m;
}

// The format hook: 16 consecutive weights of row n from column k (k a multiple of 16) as FP16. nvfp4 and fp8 decode
// exactly (post and row scales go to the accumulator); mxfp4 is exact while code * 2^(scale - 127) stays in FP16's
// range.
template <Format F>
__device__ inline void decode16(const Matrix& m, int n, int cols, int k, __half* dst) {
    if constexpr(F == Format::bf16) {
        const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(m.codes) + std::size_t(n) * cols + k;
        for(int i = 0; i < 16; ++i) dst[i] = __float2half_rn(__bfloat162float(row[i]));
    } else if constexpr(F == Format::fp8_e4m3_row) {
        const std::uint8_t* row = m.codes + std::size_t(n) * cols + k;
        for(int i = 0; i < 16; ++i) dst[i] = __float2half_rn(e4m3(row[i]));
    } else {
        const std::uint8_t* row = m.codes + std::size_t(n) * (cols / 2) + k / 2;
        float scale;
        if constexpr(F == Format::nvfp4) scale = e4m3(m.scales[std::size_t(n) * (cols / 16) + k / 16]);
        else scale = ldexpf(1.0f, int(m.scales[std::size_t(n) * (cols / 32) + k / 32]) - 127);
        for(int i = 0; i < 8; ++i) {
            const std::uint8_t byte = row[i];
            dst[2 * i] = __float2half_rn(e2m1(byte & 15u) * scale);
            dst[2 * i + 1] = __float2half_rn(e2m1(byte >> 4) * scale);
        }
    }
}

template <Format F>
__device__ inline float column_scale(const Matrix& m, int n) {
    if constexpr(F == Format::fp8_e4m3_row) return reinterpret_cast<const float*>(m.scales)[n] * m.post;
    return m.post;
}

__device__ inline int warp_sum(int v) {
    for(int d = 16; d; d /= 2) v += __shfl_down_sync(0xffffffffu, v, d);
    return v;
}

// One block per expert: its route count. The blocks also clear route_row in stripes.
__global__ void count_kernel(Routing r, Workspace w) {
    __shared__ int partial[kSortThreads / 32];
    const int e = blockIdx.x, n = r.tokens * r.top_k;
    int c = 0;
    for(int i = threadIdx.x; i < n; i += blockDim.x) c += r.ids[i] == e;
    for(int i = e * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) w.route_row[i] = -1;
    c = warp_sum(c);
    if(threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = c;
    __syncthreads();
    if(threadIdx.x == 0) {
        int total = 0;
        for(int i = 0; i < kSortThreads / 32; ++i) total += partial[i];
        w.counts[e] = total;
    }
}

__global__ void scan_kernel(Routing r, Workspace w) {
    w.offsets[0] = 0;
    w.tile_offsets[0] = 0;
    for(int e = 0; e < r.experts; ++e) {
        w.offsets[e + 1] = w.offsets[e] + w.counts[e];
        w.tile_offsets[e + 1] = w.tile_offsets[e] + (w.counts[e] + kTileRows - 1) / kTileRows;
    }
}

// One block per expert: its routes in increasing order, placed by a block-wide prefix count of each pass.
__global__ void place_kernel(Routing r, Workspace w) {
    __shared__ int warp_counts[kSortThreads / 32];
    const int e = blockIdx.x, n = r.tokens * r.top_k, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int base = w.offsets[e];
    for(int start = 0; start < n; start += blockDim.x) {
        const int i = start + threadIdx.x;
        const bool hit = i < n && r.ids[i] == e;
        const unsigned ballot = __ballot_sync(0xffffffffu, hit);
        if(lane == 0) warp_counts[warp] = __popc(ballot);
        __syncthreads();
        int before = 0, total = 0;
        for(int q = 0; q < kSortThreads / 32; ++q) {
            before += q < warp ? warp_counts[q] : 0;
            total += warp_counts[q];
        }
        if(hit) {
            const int row = base + before + __popc(ballot & ((1u << lane) - 1u));
            w.row_route[row] = i;
            w.route_row[i] = row;
        }
        base += total;
        __syncthreads();
    }
}

__global__ void gather_kernel(const __half* x, int hidden, Routing r, Workspace w, __half* xp) {
    const int row = blockIdx.x;
    if(row >= w.offsets[r.experts]) return;
    const __half* src = x + std::size_t(w.row_route[row] / r.top_k) * hidden;
    __half* dst = xp + std::size_t(row) * hidden;
    for(int h = threadIdx.x; h < hidden; h += blockDim.x) dst[h] = src[h];
}

// One 64-row tile of one expert's rows by 64 output columns. Blocks past the last tile return, so the grid is the
// host's bound (rows / 64 + experts tiles) and the launch needs no device readback.
template <Format F>
__global__ void __launch_bounds__(kProjectionThreads)
projection_kernel(const Matrix* experts, const __half* xp, int in_dim, int out_dim, Routing r, Workspace w, __half* yp) {
    using namespace nvcuda;
    __shared__ __align__(32) __half a_tile[kTileRows][kTileDepth + kPad];
    __shared__ __align__(32) __half b_tile[kTileCols][kTileDepth + kPad];
    __shared__ __align__(32) float c_tile[kTileRows][kTileCols + 4];
    const int tile = blockIdx.x;
    if(tile >= w.tile_offsets[r.experts]) return;
    int lo = 0, hi = r.experts - 1;  // the last expert whose tiles start at or before this one
    while(lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if(w.tile_offsets[mid] <= tile) lo = mid;
        else hi = mid - 1;
    }
    const int e = lo;
    const int row0 = w.offsets[e] + (tile - w.tile_offsets[e]) * kTileRows, row_end = w.offsets[e + 1];
    const int col0 = blockIdx.y * kTileCols;
    const Matrix m = experts[e];
    const int t = threadIdx.x, warp = t / 32, warp_row = (warp / 2) * 32, warp_col = (warp % 2) * 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    for(int k0 = 0; k0 < in_dim; k0 += kTileDepth) {
        {  // 16 activations of one row a thread; rows past the expert's are zero
            const int rr = t / 2, cc = (t % 2) * 16, row = row0 + rr;
            uint4* dst = reinterpret_cast<uint4*>(&a_tile[rr][cc]);
            if(row < row_end) {
                const uint4* src = reinterpret_cast<const uint4*>(xp + std::size_t(row) * in_dim + k0 + cc);
                dst[0] = src[0];
                dst[1] = src[1];
            } else {
                dst[0] = make_uint4(0, 0, 0, 0);
                dst[1] = make_uint4(0, 0, 0, 0);
            }
        }
        decode16<F>(m, col0 + t / 2, in_dim, k0 + (t % 2) * 16, &b_tile[t / 2][(t % 2) * 16]);
        __syncthreads();
        for(int kk = 0; kk < kTileDepth; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
            for(int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &a_tile[warp_row + 16 * i][kk], kTileDepth + kPad);
            for(int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &b_tile[warp_col + 16 * j][kk], kTileDepth + kPad);
            for(int i = 0; i < 2; ++i)
                for(int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&c_tile[warp_row + 16 * i][warp_col + 16 * j], acc[i][j], kTileCols + 4,
                                    wmma::mem_row_major);
    __syncthreads();
    for(int idx = t; idx < kTileRows * kTileCols; idx += kProjectionThreads) {
        const int rr = idx / kTileCols, cc = idx % kTileCols, row = row0 + rr;
        if(row < row_end)
            yp[std::size_t(row) * out_dim + col0 + cc] = __float2half_rn(c_tile[rr][cc] * column_scale<F>(m, col0 + cc));
    }
}

__global__ void swiglu_kernel(const __half* gate_up, int inter, int rows, __half* out) {
    const std::size_t n = std::size_t(rows) * inter;
    for(std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += std::size_t(gridDim.x) * blockDim.x) {
        const std::size_t row = idx / inter, i = idx % inter;
        const float g = __half2float(gate_up[row * 2 * inter + i]), u = __half2float(gate_up[row * 2 * inter + inter + i]);
        out[idx] = __float2half_rn(g / (1.0f + expf(-g)) * u);
    }
}

__global__ void combine_kernel(const __half* yp, int hidden, Routing r, Workspace w, float* out) {
    const int token = blockIdx.x;
    for(int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float acc = 0.0f;
        for(int j = 0; j < r.top_k; ++j) {
            const int route = token * r.top_k + j, row = w.route_row[route];
            if(row >= 0) acc += r.weights[route] * __half2float(yp[std::size_t(row) * hidden + h]);
        }
        out[std::size_t(token) * hidden + h] = acc;
    }
}

bool valid(const Routing& r) {
    return r.ids && r.tokens >= 0 && r.top_k > 0 && r.experts > 0;
}

bool valid(const Workspace& w) {
    return w.counts && w.offsets && w.tile_offsets && w.row_route && w.route_row;
}

} // namespace

cudaError_t enqueue_route_sort(const Routing& r, const Workspace& w, cudaStream_t stream) {
    if(!valid(r) || !valid(w)) return cudaErrorInvalidValue;
    count_kernel<<<r.experts, kSortThreads, 0, stream>>>(r, w);
    scan_kernel<<<1, 1, 0, stream>>>(r, w);
    place_kernel<<<r.experts, kSortThreads, 0, stream>>>(r, w);
    return cudaGetLastError();
}

cudaError_t enqueue_gather(const __half* x, int hidden, const Routing& r, const Workspace& w, __half* xp,
                           cudaStream_t stream) {
    if(!x || !xp || hidden < 1 || !valid(r) || !valid(w)) return cudaErrorInvalidValue;
    if(r.tokens == 0) return cudaSuccess;
    gather_kernel<<<r.tokens * r.top_k, 128, 0, stream>>>(x, hidden, r, w, xp);
    return cudaGetLastError();
}

cudaError_t enqueue_grouped_projection(Format format, const Matrix* experts, const __half* xp, int in_dim, int out_dim,
                                       const Routing& r, const Workspace& w, __half* yp, cudaStream_t stream) {
    if(!experts || !xp || !yp || in_dim < kTileDepth || in_dim % kTileDepth || out_dim < kTileCols ||
       out_dim % kTileCols || !valid(r) || !valid(w))
        return cudaErrorInvalidValue;
    if(r.tokens == 0) return cudaSuccess;
    const dim3 grid((r.tokens * r.top_k + kTileRows - 1) / kTileRows + r.experts, out_dim / kTileCols);
    switch(format) {
    case Format::bf16:
        projection_kernel<Format::bf16><<<grid, kProjectionThreads, 0, stream>>>(experts, xp, in_dim, out_dim, r, w, yp);
        break;
    case Format::fp8_e4m3_row:
        projection_kernel<Format::fp8_e4m3_row><<<grid, kProjectionThreads, 0, stream>>>(experts, xp, in_dim, out_dim, r, w, yp);
        break;
    case Format::nvfp4:
        projection_kernel<Format::nvfp4><<<grid, kProjectionThreads, 0, stream>>>(experts, xp, in_dim, out_dim, r, w, yp);
        break;
    case Format::mxfp4:
        projection_kernel<Format::mxfp4><<<grid, kProjectionThreads, 0, stream>>>(experts, xp, in_dim, out_dim, r, w, yp);
        break;
    default:
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

cudaError_t enqueue_swiglu(const __half* gate_up, int inter, int rows, __half* out, cudaStream_t stream) {
    if(!gate_up || !out || inter < 1 || rows < 0) return cudaErrorInvalidValue;
    if(rows == 0) return cudaSuccess;
    const std::size_t n = std::size_t(rows) * inter;
    const int blocks = int(n / 256 + 1 < 65535 ? n / 256 + 1 : 65535);
    swiglu_kernel<<<blocks, 256, 0, stream>>>(gate_up, inter, rows, out);
    return cudaGetLastError();
}

cudaError_t enqueue_combine(const __half* yp, int hidden, const Routing& r, const Workspace& w, float* out,
                            cudaStream_t stream) {
    if(!yp || !out || hidden < 1 || !valid(r) || !r.weights || !valid(w)) return cudaErrorInvalidValue;
    if(r.tokens == 0) return cudaSuccess;
    combine_kernel<<<r.tokens, 256, 0, stream>>>(yp, hidden, r, w, out);
    return cudaGetLastError();
}

} // namespace crit::techniques::streamed_prefill
