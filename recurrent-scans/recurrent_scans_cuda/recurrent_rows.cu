// Pool kernels of recurrent_rows.h. FP32 SIMT throughout; see the header for the layouts and the commit strategies.
#include "recurrent_rows.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace crit::recurrent_kernels {
namespace {

constexpr int kRowThreads = 256;    // 8 warps x 4 value columns
constexpr int kRowColumns = 32;     // value columns per block
constexpr int kMaxDecays = kRowThreads;
constexpr int kPoolConvThreads = 128, kPoolConvSegment = 32, kPoolMaxConv = 8;

template <class T> __device__ __forceinline__ float as_float(T x);
template <> __device__ __forceinline__ float as_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float as_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <> __device__ __forceinline__ float as_float<__half>(__half x) { return __half2float(x); }
template <class T> __device__ __forceinline__ T of_float(float x);
template <> __device__ __forceinline__ float of_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __nv_bfloat16 of_float<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }
template <> __device__ __forceinline__ __half of_float<__half>(float x) { return __float2half_rn(x); }

// A token's decay e^l: FP64 exp rounded once to FP32 (recurrent_kernels.cu's decay_of), computed once per token per
// block so every lane uses the same value.
__device__ __forceinline__ float pool_decay(float l) { return __double2float_rn(exp(double(l))); }

// The one update expression (recurrent_kernels.cu's delta_update).
__device__ __forceinline__ float pool_update(float s, float e, float k, float delta) { return fmaf(k, delta, s * e); }

// recurrent_kernels.cu's warp_sum8: the 8 dot products of a warp's 4 columns (k . S, q . S) reduced and broadcast.
__device__ __forceinline__ void pool_sum8(float (&x)[8]) {
    const int lane = threadIdx.x & 31;
    float a[4], b[2], c;
    {
        const bool hi = lane & 16;
#pragma unroll
        for(int i = 0; i < 4; ++i) a[i] = (hi ? x[i + 4] : x[i]) + __shfl_xor_sync(0xffffffffu, hi ? x[i] : x[i + 4], 16);
    }
    {
        const bool hi = lane & 8;
#pragma unroll
        for(int i = 0; i < 2; ++i) b[i] = (hi ? a[i + 2] : a[i]) + __shfl_xor_sync(0xffffffffu, hi ? a[i] : a[i + 2], 8);
    }
    {
        const bool hi = lane & 4;
        c = (hi ? b[1] : b[0]) + __shfl_xor_sync(0xffffffffu, hi ? b[0] : b[1], 4);
    }
    c += __shfl_xor_sync(0xffffffffu, c, 2);
    c += __shfl_xor_sync(0xffffffffu, c, 1);
#pragma unroll
    for(int i = 0; i < 8; ++i) x[i] = __shfl_sync(0xffffffffu, c, 4 * i);
}

// The same scheme for 4 values (|k|^2, |q|^2, q . k and a zero): distances 16, 8, then 4, 2, 1; value i from lane 8 i.
__device__ __forceinline__ void pool_sum4(float (&x)[4]) {
    const int lane = threadIdx.x & 31;
    float a[2], c;
    {
        const bool hi = lane & 16;
#pragma unroll
        for(int i = 0; i < 2; ++i) a[i] = (hi ? x[i + 2] : x[i]) + __shfl_xor_sync(0xffffffffu, hi ? x[i] : x[i + 2], 16);
    }
    {
        const bool hi = lane & 8;
        c = (hi ? a[1] : a[0]) + __shfl_xor_sync(0xffffffffu, hi ? a[0] : a[1], 8);
    }
    c += __shfl_xor_sync(0xffffffffu, c, 4);
    c += __shfl_xor_sync(0xffffffffu, c, 2);
    c += __shfl_xor_sync(0xffffffffu, c, 1);
#pragma unroll
    for(int i = 0; i < 4; ++i) x[i] = __shfl_sync(0xffffffffu, c, 8 * i);
}

struct Row { int first, count, sequence, kind, slot; };

__device__ __forceinline__ Row row_of(const RowSource& s, int r) {
    Row w;
    if(s.meta) {
        const std::int32_t* m = s.meta + 4 * r;
        w.first = m[0];
        w.count = m[1];
        w.sequence = m[2];
        w.kind = m[3];
        w.slot = r;
    } else {   // a commit reads no tokens: starts may be absent
        w.first = s.starts ? s.starts[r] : 0;
        w.count = s.starts ? s.starts[r + 1] - w.first : 0;
        w.slot = int(s.slots[r]);
        w.sequence = s.table[w.slot];
        w.kind = s.kind;
    }
    return w;
}

// One stash half: key [capacity][k_heads][dk], decay [capacity][v_heads], delta [capacity][v_heads][dv].
struct Half { float* key; float* decay; float* delta; };
__host__ __device__ __forceinline__ std::size_t half_floats(const DeltaShape& d, int capacity) {
    return std::size_t(capacity) * (std::size_t(d.k_heads) * d.dk + d.v_heads + std::size_t(d.v_heads) * d.dv);
}
__device__ __forceinline__ Half half_at(const DeltaShape& d, float* stash, int parity, int capacity) {
    Half h;
    h.key = stash + std::size_t(parity) * half_floats(d, capacity);
    h.decay = h.key + std::size_t(capacity) * d.k_heads * d.dk;
    h.delta = h.decay + std::size_t(capacity) * d.v_heads;
    return h;
}

// A lane's tile: key rows lane * R .. + R - 1 of value columns j0 .. j0 + 3 of one head's [dk][dv] state.
template <int R>
__device__ __forceinline__ void tile_load(float (&S)[R][4], const float* head, int dk, int dv, int lane, int j0) {
    const bool wide = (dv & 3) == 0 && j0 < dv && (reinterpret_cast<std::uintptr_t>(head) & 15u) == 0;
#pragma unroll
    for(int r = 0; r < R; ++r) {
        const int x = lane * R + r;
        if(x < dk && wide) {
            const float4 v = *reinterpret_cast<const float4*>(head + std::size_t(x) * dv + j0);
            S[r][0] = v.x; S[r][1] = v.y; S[r][2] = v.z; S[r][3] = v.w;
        } else {
#pragma unroll
            for(int j = 0; j < 4; ++j) S[r][j] = x < dk && j0 + j < dv ? head[std::size_t(x) * dv + j0 + j] : 0.f;
        }
    }
}
template <int R>
__device__ __forceinline__ void tile_store(float* head, const float (&S)[R][4], int dk, int dv, int lane, int j0) {
    const bool wide = (dv & 3) == 0 && j0 < dv && (reinterpret_cast<std::uintptr_t>(head) & 15u) == 0;
#pragma unroll
    for(int r = 0; r < R; ++r) {
        const int x = lane * R + r;
        if(x >= dk) continue;
        if(wide) {
            *reinterpret_cast<float4*>(head + std::size_t(x) * dv + j0) = make_float4(S[r][0], S[r][1], S[r][2], S[r][3]);
        } else {
#pragma unroll
            for(int j = 0; j < 4; ++j)
                if(j0 + j < dv) head[std::size_t(x) * dv + j0 + j] = S[r][j];
        }
    }
}

// Apply `n` stashed steps of value head h (key head kh) to the tile.
template <int R>
__device__ __forceinline__ void tile_replay(float (&S)[R][4], const DeltaShape& d, const Half& from, int n, int h, int kh,
                                            int lane, int j0) {
    for(int t = 0; t < n; ++t) {
        const float e = from.decay[std::size_t(t) * d.v_heads + h];
        const float* kr = from.key + (std::size_t(t) * d.k_heads + kh) * d.dk;
        const float* dr = from.delta + (std::size_t(t) * d.v_heads + h) * d.dv;
        float k[R], delta[4];
#pragma unroll
        for(int r = 0; r < R; ++r) k[r] = lane * R + r < d.dk ? kr[lane * R + r] : 0.f;
#pragma unroll
        for(int j = 0; j < 4; ++j) delta[j] = j0 + j < d.dv ? dr[j0 + j] : 0.f;
#pragma unroll
        for(int r = 0; r < R; ++r)
#pragma unroll
            for(int j = 0; j < 4; ++j) S[r][j] = pool_update(S[r][j], e, k[r], delta[j]);
    }
}

// The inputs of one token for a lane: its key rows of q and k, its warp's 4 value columns of v, and beta.
template <int R>
struct Token { float k[R], q[R], v[4], b; };

template <int R, class T>
__device__ __forceinline__ void token_load(Token<R>& u, const DeltaShape& d, const RowInputs& in, std::int64_t t, int h, int kh,
                                           int lane, int j0) {
    const T* kr = static_cast<const T*>(in.k) + t * in.k_row + kh * in.k_head;
    const T* qr = static_cast<const T*>(in.q) + t * in.q_row + kh * in.q_head;
    const T* vr = static_cast<const T*>(in.v) + t * in.v_row + h * in.v_head;
#pragma unroll
    for(int r = 0; r < R; ++r) {
        const int x = lane * R + r;
        u.k[r] = x < d.dk ? as_float<T>(kr[x]) : 0.f;
        u.q[r] = x < d.dk ? as_float<T>(qr[x]) : 0.f;
    }
#pragma unroll
    for(int j = 0; j < 4; ++j) u.v[j] = j0 + j < d.dv ? as_float<T>(vr[j0 + j]) : 0.f;
    u.b = in.beta[t * in.beta_row + h];
}

// One token: the dot products and norms reduced once, delta, the output and the update. kn receives the normalized
// keys the update used.
template <int R>
__device__ __forceinline__ void token_step(float (&S)[R][4], const Token<R>& u, const RowInputs& in, float e, float (&kn)[R],
                                           float (&delta)[4], float (&o)[4]) {
    float a[4] = {0.f, 0.f, 0.f, 0.f}, x[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for(int r = 0; r < R; ++r) {
        a[0] = fmaf(u.k[r], u.k[r], a[0]);
        a[1] = fmaf(u.q[r], u.q[r], a[1]);
        a[2] = fmaf(u.q[r], u.k[r], a[2]);
#pragma unroll
        for(int j = 0; j < 4; ++j) {
            x[j] = fmaf(u.k[r], S[r][j], x[j]);
            x[4 + j] = fmaf(u.q[r], S[r][j], x[4 + j]);
        }
    }
    pool_sum8(x);
    pool_sum4(a);
    const float ik = in.normalize_keys ? 1.f / sqrtf(a[0] + in.epsilon) : 1.f;
    const float iq = (in.normalize_queries ? 1.f / sqrtf(a[1] + in.epsilon) : 1.f) * in.scale;
    const float qk = a[2] * ik * iq;
#pragma unroll
    for(int j = 0; j < 4; ++j) {
        delta[j] = u.b * (u.v[j] - e * (x[j] * ik));
        o[j] = fmaf(e, x[4 + j] * iq, qk * delta[j]);
    }
#pragma unroll
    for(int r = 0; r < R; ++r) {
        kn[r] = u.k[r] * ik;
#pragma unroll
        for(int j = 0; j < 4; ++j) S[r][j] = pool_update(S[r][j], e, kn[r], delta[j]);
    }
}

// Rows of one layer, block per (value head, 32 value columns, row).
template <int R, class T>
__global__ void __launch_bounds__(kRowThreads) pool_delta_kernel(DeltaShape d, RowInputs in, PoolDelta p, RowSource src) {
    const Row w = row_of(src, blockIdx.z);
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ratio = d.v_heads / d.k_heads, kh = h / ratio;
    const int j0 = blockIdx.y * kRowColumns + warp * 4;
    if(w.sequence < 0) {   // a row of no sequence (an unbound round slot): zero outputs, no state read or written
        if(lane == 0)
            for(int t = 0; t < w.count; ++t)
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    if(j0 + j < d.dv) in.out[(w.first + std::int64_t(t)) * in.out_row + std::int64_t(h) * d.dv + j0 + j] = 0.f;
        return;
    }
    const int blocks = d.v_heads * gridDim.y, slot = h * gridDim.y + blockIdx.y;
    const std::int64_t S_size = std::int64_t(d.v_heads) * d.dk * d.dv, head = std::int64_t(h) * d.dk * d.dv;
    float* live = p.states + w.sequence * p.state_sequence + head;
    float* stash = p.stash + w.sequence * p.stash_sequence;
    std::int32_t* control = p.control + w.sequence * p.control_sequence;
    __shared__ int s_pending, s_parity;
    __shared__ float decay[kMaxDecays];
    float S[R][4];
    tile_load<R>(S, live, d.dk, d.dv, lane, j0);
    if(threadIdx.x == 0) {
        s_pending = min(max(control[slot], 0), p.capacity);
        s_parity = control[blocks + slot] & 1;
    }
    __syncthreads();
    const int pending = s_pending, parity = s_parity;
    const bool stashes = (w.kind == kStash || w.kind == kRecompute) && w.count > 0;
    if(threadIdx.x == 0) {   // consumed; a stashing row writes the other half and flips the parity
        control[slot] = 0;
        if(stashes) control[blocks + slot] = parity ^ 1;
    }
    if(pending) {
        tile_replay<R>(S, d, half_at(d, stash, parity, p.capacity), pending, h, kh, lane, j0);
        tile_store<R>(live, S, d.dk, d.dv, lane, j0);
    }
    if(w.kind == kFlush || w.count <= 0) return;
    float* scratch = p.scratch ? p.scratch + std::int64_t(w.slot) * p.scratch_slot : nullptr;
    if(w.kind == kCheckpoints) tile_store<R>(scratch + head, S, d.dk, d.dv, lane, j0);
    if(w.kind == kRecompute) tile_store<R>(scratch + head, S, d.dk, d.dv, lane, j0);
    const Half next = half_at(d, stash, parity ^ 1, p.capacity);
    const bool writes_key = blockIdx.y == 0 && h % ratio == 0 && warp == 0, writes_decay = blockIdx.y == 0 && threadIdx.x == 0;
    Token<R> u;
    token_load<R, T>(u, d, in, w.first, h, kh, lane, j0);
    for(int t0 = 0; t0 < w.count; t0 += kMaxDecays) {
        __syncthreads();   // the previous pass's readers of decay are done
        if(t0 + int(threadIdx.x) < w.count) decay[threadIdx.x] = pool_decay(in.log_decay[(w.first + t0 + std::int64_t(threadIdx.x)) * in.gate_row + h]);
        __syncthreads();
        for(int t = t0; t < min(w.count, t0 + kMaxDecays); ++t) {
            const Token<R> now = u;
            if(t + 1 < w.count) token_load<R, T>(u, d, in, w.first + t + 1, h, kh, lane, j0);   // in flight meanwhile
            const float e = decay[t - t0];
            float kn[R], delta[4], o[4];
            token_step<R>(S, now, in, e, kn, delta, o);
            const std::int64_t row = w.first + t;
            if(lane == 0)
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    if(j0 + j < d.dv) in.out[row * in.out_row + std::int64_t(h) * d.dv + j0 + j] = o[j];
            if(stashes && t < p.capacity) {
                if(lane == 0)
#pragma unroll
                    for(int j = 0; j < 4; ++j)
                        if(j0 + j < d.dv) next.delta[(std::size_t(t) * d.v_heads + h) * d.dv + j0 + j] = delta[j];
                if(writes_key)
#pragma unroll
                    for(int r = 0; r < R; ++r)
                        if(lane * R + r < d.dk) next.key[(std::size_t(t) * d.k_heads + kh) * d.dk + lane * R + r] = kn[r];
                if(writes_decay) next.decay[std::size_t(t) * d.v_heads + h] = e;
            }
            if(w.kind == kCheckpoints && t < p.capacity)
                tile_store<R>(scratch + (t + 1) * S_size + head, S, d.dk, d.dv, lane, j0);
        }
    }
    if(w.kind == kFinal) tile_store<R>(scratch + head, S, d.dk, d.dv, lane, j0);
}

// kReplay and kFlushPending over every layer: block per (value head, 32 value columns, row * layers + layer).
template <int R>
__global__ void __launch_bounds__(kRowThreads) pool_replay_kernel(DeltaShape d, PoolDelta p, RowSource src, int layers,
                                                                  const std::int64_t* accepted, int columns, int mode) {
    const int r = blockIdx.z / layers, layer = blockIdx.z % layers;
    const Row w = row_of(src, r);
    if(w.sequence < 0) return;
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kh = h / (d.v_heads / d.k_heads), j0 = blockIdx.y * kRowColumns + warp * 4;
    const int blocks = d.v_heads * gridDim.y, slot = h * gridDim.y + blockIdx.y;
    const std::int64_t head = std::int64_t(h) * d.dk * d.dv;
    float* live = p.states + layer * p.state_layer + w.sequence * p.state_sequence + head;
    float* stash = p.stash + layer * p.stash_layer + w.sequence * p.stash_sequence;
    std::int32_t* control = p.control + layer * p.control_layer + w.sequence * p.control_sequence;
    __shared__ int s_n, s_parity;
    if(threadIdx.x == 0) {
        s_parity = control[blocks + slot] & 1;
        if(mode == kFlushPending) {
            s_n = min(max(control[slot], 0), p.capacity);
        } else {
            const std::int64_t n = accepted[r];
            s_n = int(n < 0 ? 0 : n > columns ? columns : n);
        }
    }
    __syncthreads();
    const int n = s_n;
    if(mode == kFlushPending && threadIdx.x == 0) control[slot] = 0;
    if(mode == kFlushPending && !n) return;
    float S[R][4];
    const float* from = mode == kReplay ? p.scratch + layer * p.scratch_layer + std::int64_t(w.slot) * p.scratch_slot + head : live;
    tile_load<R>(S, from, d.dk, d.dv, lane, j0);
    tile_replay<R>(S, d, half_at(d, stash, s_parity, p.capacity), n, h, kh, lane, j0);
    tile_store<R>(live, S, d.dk, d.dv, lane, j0);
}

template <int R>
cudaError_t replay_launch(const DeltaShape& d, const PoolDelta& p, const RowSource& s, int layers, const std::int64_t* accepted,
                          int columns, int mode, cudaStream_t stream) {
    const dim3 grid(d.v_heads, (d.dv + kRowColumns - 1) / kRowColumns, s.rows * layers);
    pool_replay_kernel<R><<<grid, kRowThreads, 0, stream>>>(d, p, s, layers, accepted, columns, mode);
    return cudaGetLastError();
}

template <int R, class T>
cudaError_t rows_launch(const DeltaShape& d, const RowInputs& in, const PoolDelta& p, const RowSource& s, cudaStream_t stream) {
    const dim3 grid(d.v_heads, (d.dv + kRowColumns - 1) / kRowColumns, s.rows);
    pool_delta_kernel<R, T><<<grid, kRowThreads, 0, stream>>>(d, in, p, s);
    return cudaGetLastError();
}

template <int R>
cudaError_t rows_dtype(const DeltaShape& d, const RowInputs& in, const PoolDelta& p, const RowSource& s, cudaStream_t stream) {
    switch(in.dtype) {
    case Scalar::bf16: return rows_launch<R, __nv_bfloat16>(d, in, p, s, stream);
    case Scalar::f16: return rows_launch<R, __half>(d, in, p, s, stream);
    default: return rows_launch<R, float>(d, in, p, s, stream);
    }
}

int lane_rows(int dk) { return dk <= 32 ? 1 : dk <= 64 ? 2 : dk <= 128 ? 4 : 8; }

bool valid_source(const RowSource& s, bool tokens) {
    return s.rows >= 0 && (s.meta || (s.slots && s.table && (s.starts || !tokens))) && s.rows <= 65535;
}
bool valid_pool(const DeltaShape& d, const PoolDelta& p) {
    return pool_delta_supported(d) && p.states && p.stash && p.control && p.capacity >= 1;
}

// ---------------------------------------------------------------------------------------------- convolution
template <int W>
__device__ __forceinline__ float pool_conv_value(const float (&win)[W - 1], const float (&w)[W], float b, float xt, bool act) {
    float acc = b;
#pragma unroll
    for(int i = 0; i < W - 1; ++i) acc = fmaf(w[i], win[i], acc);
    acc = fmaf(w[W - 1], xt, acc);
    return act ? acc / (1.f + expf(-acc)) : acc;
}
template <int W>
__device__ __forceinline__ void pool_conv_shift(float (&win)[W - 1], float xt) {
#pragma unroll
    for(int i = 0; i < W - 2; ++i) win[i] = win[i + 1];
    win[W - 2] = xt;
}

template <int W, class T>
__global__ void __launch_bounds__(kPoolConvThreads) pool_conv_kernel(ConvInputs c, PoolConv p, RowSource src, int layers) {
    const int r = blockIdx.y / layers, layer = blockIdx.y % layers;
    const Row w = row_of(src, r);
    const int ch = blockIdx.x * kPoolConvThreads + threadIdx.x;
    if(w.sequence < 0) {   // a row of no sequence (an unbound round slot): zero outputs, no window read or written
        if(ch < c.channels && w.kind != kFlush)
            for(int t = 0; t < w.count; ++t) static_cast<T*>(c.out)[(w.first + std::int64_t(t)) * c.out_row + ch] = of_float<T>(0.f);
        return;
    }
    std::int32_t* pending = p.pending + layer * p.pending_layer + w.sequence * p.pending_sequence + blockIdx.x;
    __shared__ int s_n;
    if(threadIdx.x == 0) s_n = min(max(*pending, 0), p.capacity);
    __syncthreads();
    if(threadIdx.x == 0) *pending = 0;
    if(ch >= c.channels) return;
    const int n = s_n;
    float* window = p.window + layer * p.window_layer + w.sequence * p.window_sequence + std::int64_t(ch) * (W - 1);
    const float* stash_in = p.stash + layer * p.stash_layer + w.sequence * p.stash_sequence;
    float win[W - 1];
#pragma unroll
    for(int i = 0; i < W - 1; ++i) win[i] = window[i];
    for(int t = 0; t < n; ++t) pool_conv_shift<W>(win, stash_in[std::int64_t(t) * c.channels + ch]);
    if(n)
#pragma unroll
        for(int i = 0; i < W - 1; ++i) window[i] = win[i];
    if(w.kind == kFlush || w.count <= 0) return;
    float wt[W];
#pragma unroll
    for(int i = 0; i < W; ++i) wt[i] = c.weight[std::int64_t(ch) * W + i];
    const float b = c.bias ? c.bias[ch] : 0.f;
    float* stash_out = p.stash + layer * p.stash_layer + w.sequence * p.stash_sequence;
    for(int t = 0; t < w.count; ++t) {
        const std::int64_t row = w.first + t;
        const float xt = as_float<T>(static_cast<const T*>(c.x)[row * c.x_row + ch]);
        static_cast<T*>(c.out)[row * c.out_row + ch] = of_float<T>(pool_conv_value<W>(win, wt, b, xt, c.silu));
        pool_conv_shift<W>(win, xt);
        if(w.kind == kStash && t < p.capacity) stash_out[std::int64_t(t) * c.channels + ch] = xt;
    }
    if(w.kind == kFinal) {
        float* to = p.scratch + std::int64_t(w.slot) * p.scratch_slot + std::int64_t(ch) * (W - 1);
#pragma unroll
        for(int i = 0; i < W - 1; ++i) to[i] = win[i];
    }
}

// One long row, thread per (channel, kPoolConvSegment rows), recurrent_kernels.cu's conv_segments_kernel over pool windows.
template <int W, class T>
__global__ void __launch_bounds__(kPoolConvThreads) conv_segments_kernel(ConvInputs c, const float* window_in, float* window_out,
                                                                         int rows) {
    const int ch = blockIdx.x * kPoolConvThreads + threadIdx.x, t0 = blockIdx.y * kPoolConvSegment;
    if(ch >= c.channels) return;
    const T* x = static_cast<const T*>(c.x);
    const int n = min(kPoolConvSegment, rows - t0);
    float win[W - 1], wt[W];
#pragma unroll
    for(int i = 0; i < W - 1; ++i)
        win[i] = t0 ? as_float<T>(x[std::int64_t(t0 - (W - 1) + i) * c.x_row + ch]) : window_in[std::int64_t(ch) * (W - 1) + i];
#pragma unroll
    for(int i = 0; i < W; ++i) wt[i] = c.weight[std::int64_t(ch) * W + i];
    const float b = c.bias ? c.bias[ch] : 0.f;
    for(int i = 0; i < n; ++i) {
        const int t = t0 + i;
        const float xt = as_float<T>(x[std::int64_t(t) * c.x_row + ch]);
        static_cast<T*>(c.out)[std::int64_t(t) * c.out_row + ch] = of_float<T>(pool_conv_value<W>(win, wt, b, xt, c.silu));
        pool_conv_shift<W>(win, xt);
    }
    if(blockIdx.y == 0)
#pragma unroll
        for(int i = 0; i < W - 1; ++i) {   // the last W - 1 inputs, from the window where the row is shorter
            const int at = rows - (W - 1) + i;
            window_out[std::int64_t(ch) * (W - 1) + i] =
                at >= 0 ? as_float<T>(x[std::int64_t(at) * c.x_row + ch]) : window_in[std::int64_t(ch) * (W - 1) + (W - 1) + at];
        }
}

template <int W, class T>
cudaError_t conv_launch(const ConvInputs& c, const PoolConv* p, const RowSource* s, int layers, const float* in, float* out,
                        int rows, cudaStream_t stream) {
    const int blocks = (c.channels + kPoolConvThreads - 1) / kPoolConvThreads;
    if(p) pool_conv_kernel<W, T><<<dim3(blocks, s->rows * layers), kPoolConvThreads, 0, stream>>>(c, *p, *s, layers);
    else conv_segments_kernel<W, T><<<dim3(blocks, (rows + kPoolConvSegment - 1) / kPoolConvSegment), kPoolConvThreads, 0, stream>>>(c, in, out, rows);
    return cudaGetLastError();
}
template <class T>
cudaError_t conv_width(const ConvInputs& c, const PoolConv* p, const RowSource* s, int layers, const float* in, float* out,
                       int rows, cudaStream_t stream) {
    switch(c.width) {
    case 2: return conv_launch<2, T>(c, p, s, layers, in, out, rows, stream);
    case 3: return conv_launch<3, T>(c, p, s, layers, in, out, rows, stream);
    case 4: return conv_launch<4, T>(c, p, s, layers, in, out, rows, stream);
    case 5: return conv_launch<5, T>(c, p, s, layers, in, out, rows, stream);
    case 6: return conv_launch<6, T>(c, p, s, layers, in, out, rows, stream);
    case 7: return conv_launch<7, T>(c, p, s, layers, in, out, rows, stream);
    default: return conv_launch<8, T>(c, p, s, layers, in, out, rows, stream);
    }
}
cudaError_t conv_dispatch(const ConvInputs& c, const PoolConv* p, const RowSource* s, int layers, const float* in, float* out,
                          int rows, cudaStream_t stream) {
    switch(c.dtype) {
    case Scalar::bf16: return conv_width<__nv_bfloat16>(c, p, s, layers, in, out, rows, stream);
    case Scalar::f16: return conv_width<__half>(c, p, s, layers, in, out, rows, stream);
    default: return conv_width<float>(c, p, s, layers, in, out, rows, stream);
    }
}

}  // namespace

bool pool_delta_supported(const DeltaShape& d) {
    return d.k_heads > 0 && d.v_heads > 0 && d.v_heads % d.k_heads == 0 && d.dk >= 1 && d.dk <= 256 && d.dv >= 1 &&
           d.v_heads <= 65535 && (d.dv + kRowColumns - 1) / kRowColumns <= 65535;
}

int pool_delta_blocks(const DeltaShape& d) { return d.v_heads * ((d.dv + kRowColumns - 1) / kRowColumns); }

std::size_t pool_delta_half_floats(const DeltaShape& d, int capacity) { return half_floats(d, capacity); }

cudaError_t enqueue_pool_delta(const DeltaShape& d, const RowInputs& in, const PoolDelta& p, const RowSource& s, cudaStream_t stream) {
    if(!valid_pool(d, p) || !valid_source(s, true) || !in.q || !in.k || !in.v || !in.log_decay || !in.beta || !in.out)
        return cudaErrorInvalidValue;
    if(!s.rows) return cudaSuccess;
    if(!s.meta && (s.kind == kCheckpoints || s.kind == kRecompute || s.kind == kFinal) && !p.scratch) return cudaErrorInvalidValue;
    switch(lane_rows(d.dk)) {
    case 1: return rows_dtype<1>(d, in, p, s, stream);
    case 2: return rows_dtype<2>(d, in, p, s, stream);
    case 4: return rows_dtype<4>(d, in, p, s, stream);
    default: return rows_dtype<8>(d, in, p, s, stream);
    }
}

cudaError_t enqueue_pool_delta_replay(const DeltaShape& d, const PoolDelta& p, const RowSource& s, int layers,
                                      const std::int64_t* accepted, int columns, CommitMode mode, cudaStream_t stream) {
    if(!valid_pool(d, p) || !valid_source(s, false) || layers < 1 || columns < 0) return cudaErrorInvalidValue;
    if(mode != kReplay && mode != kFlushPending) return cudaErrorInvalidValue;
    if(mode == kReplay && (!accepted || !p.scratch || columns > p.capacity)) return cudaErrorInvalidValue;
    if(!s.rows) return cudaSuccess;
    switch(lane_rows(d.dk)) {
    case 1: return replay_launch<1>(d, p, s, layers, accepted, columns, mode, stream);
    case 2: return replay_launch<2>(d, p, s, layers, accepted, columns, mode, stream);
    case 4: return replay_launch<4>(d, p, s, layers, accepted, columns, mode, stream);
    default: return replay_launch<8>(d, p, s, layers, accepted, columns, mode, stream);
    }
}

int pool_conv_blocks(int channels) { return channels < 1 ? 0 : (channels + kPoolConvThreads - 1) / kPoolConvThreads; }

cudaError_t enqueue_pool_conv(const ConvInputs& c, const PoolConv& p, const RowSource& s, int layers, cudaStream_t stream) {
    if(!valid_source(s, true) || layers < 1 || c.channels < 1 || c.width < 2 || c.width > kPoolMaxConv || !c.weight || !p.window ||
       !p.stash || !p.pending || p.capacity < 1 || std::int64_t(s.rows) * layers > 65535)
        return cudaErrorInvalidValue;
    if(!s.rows) return cudaSuccess;
    return conv_dispatch(c, &p, &s, layers, nullptr, nullptr, 0, stream);
}

cudaError_t enqueue_conv_segments(const ConvInputs& c, const float* window_in, float* window_out, int rows, cudaStream_t stream) {
    if(rows < 0 || c.channels < 1 || c.width < 2 || c.width > kPoolMaxConv || !c.x || !c.out || !c.weight || !window_in ||
       !window_out || c.out == c.x || (rows + kPoolConvSegment - 1) / kPoolConvSegment > 65535)
        return cudaErrorInvalidValue;
    if(!rows) return cudaSuccess;
    return conv_dispatch(c, nullptr, nullptr, 1, window_in, window_out, rows, stream);
}

}  // namespace crit::recurrent_kernels
