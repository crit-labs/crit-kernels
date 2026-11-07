// Top-k routing, a warp per token row (topk_routing.h). Each lane holds the logits of experts lane, lane + 32, ...; the
// scores, the selection by repeated warp argmax (the larger value, then the lower id) and the weights stay in
// registers, so a row costs one read of its logits and one write of its picks.
#include "topk_routing.h"

#include <math_constants.h>

namespace crit::techniques::topk_routing {
namespace {

constexpr int kWarps = 4;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for(int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for(int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

template <int PER>
__global__ void __launch_bounds__(kWarps * 32) route_kernel(const float* __restrict__ logits, int ld, int tokens,
                                                            int experts, int top, const float* __restrict__ bias,
                                                            Rule rule, long long* __restrict__ ids,
                                                            float* __restrict__ weights) {
    const int row = blockIdx.x * kWarps + static_cast<int>(threadIdx.x) / 32, lane = threadIdx.x & 31;
    if(row >= tokens) return;
    const float* l = logits + static_cast<long long>(row) * ld;
    float x[PER], score[PER], choice[PER];
#pragma unroll
    for(int i = 0; i < PER; ++i) {
        const int e = lane + 32 * i;
        x[i] = e < experts ? l[e] : -CUDART_INF_F;
    }
    if(rule.before && rule.softmax) {   // a NaN anywhere in the row makes every score NaN, as a softmax does
        float m = -CUDART_INF_F;
#pragma unroll
        for(int i = 0; i < PER; ++i) m = fmaxf(m, x[i]);
        m = warp_max(m);
        float s = 0.0f;
#pragma unroll
        for(int i = 0; i < PER; ++i) {
            score[i] = lane + 32 * i < experts ? expf(x[i] - m) : 0.0f;
            s += score[i];
        }
        s = warp_sum(s);
#pragma unroll
        for(int i = 0; i < PER; ++i) score[i] = score[i] / s;
    } else if(rule.before) {
#pragma unroll
        for(int i = 0; i < PER; ++i) score[i] = 1.0f / (1.0f + expf(-x[i]));
    } else {
#pragma unroll
        for(int i = 0; i < PER; ++i) score[i] = x[i];
    }
#pragma unroll
    for(int i = 0; i < PER; ++i) {
        const int e = lane + 32 * i;
        float c = score[i] + (bias && e < experts ? bias[e] : 0.0f);
        choice[i] = (e < experts && !isnan(c)) ? c : -CUDART_INF_F;
    }
    unsigned taken = 0;
    int picked = -1;          // lane k keeps pick k
    float picked_value = 0.0f, picked_logit = 0.0f;
    for(int k = 0; k < top; ++k) {
        float best = -CUDART_INF_F;
        int at = 0x7fffffff;
#pragma unroll
        for(int i = 0; i < PER; ++i) {
            const int e = lane + 32 * i;
            if((taken >> i) & 1u) continue;
            const int key = e < experts ? e : experts + e;   // past the experts: never before a real one
            if(choice[i] > best || (choice[i] == best && key < at)) {
                best = choice[i];
                at = key;
            }
        }
#pragma unroll
        for(int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oa = __shfl_xor_sync(0xffffffffu, at, o);
            if(ob > best || (ob == best && oa < at)) {
                best = ob;
                at = oa;
            }
        }
        float s = 0.0f, lx = 0.0f;
#pragma unroll
        for(int i = 0; i < PER; ++i)
            if(lane + 32 * i == at) {
                s = score[i];
                lx = x[i];
                taken |= 1u << i;
            }
        const int owner = at & 31;
        s = __shfl_sync(0xffffffffu, s, owner);
        lx = __shfl_sync(0xffffffffu, lx, owner);
        if(lane == k) {
            picked = at;
            picked_value = s;
            picked_logit = lx;
        }
    }
    // Weights in selection order: the picked scores, or the picked logits' softmax / sigmoid.
    float w = picked_value;
    if(!rule.before) {
        if(rule.softmax) {
            float m = lane < top ? picked_logit : -CUDART_INF_F;
            m = warp_max(m);
            float e = lane < top ? expf(picked_logit - m) : 0.0f;
            float sum = 0.0f;
            for(int k = 0; k < top; ++k) sum += __shfl_sync(0xffffffffu, e, k);
            w = e / sum;
        } else {
            w = 1.0f / (1.0f + expf(-picked_logit));
        }
    }
    if(rule.renormalize) {
        float sum = 0.0f;
        for(int k = 0; k < top; ++k) sum += __shfl_sync(0xffffffffu, w, k);
        w = w / fmaxf(sum + rule.epsilon, rule.floor);
    }
    w = w * rule.scaling;
    if(lane < top) {
        ids[static_cast<long long>(row) * top + lane] = picked;
        weights[static_cast<long long>(row) * top + lane] = isnan(w) ? 0.0f : w;
    }
}

template <int PER>
void launch(const float* logits, int ld, int tokens, int experts, int top, const float* bias, const Rule& rule,
            long long* ids, float* weights, cudaStream_t stream) {
    route_kernel<PER><<<(tokens + kWarps - 1) / kWarps, kWarps * 32, 0, stream>>>(logits, ld, tokens, experts, top, bias,
                                                                                  rule, ids, weights);
}

}  // namespace

cudaError_t enqueue_route(const float* logits, int ld, int tokens, int experts, int top, const float* bias,
                          const Rule& rule, long long* ids, float* weights, cudaStream_t stream) {
    if(!logits || !ids || !weights || tokens < 0 || experts < 1 || experts > kMaxExperts || top < 1 || top > experts ||
       top > kMaxTop || ld < experts)
        return cudaErrorInvalidValue;
    if(tokens == 0) return cudaSuccess;
    if(experts <= 32) launch<1>(logits, ld, tokens, experts, top, bias, rule, ids, weights, stream);
    else if(experts <= 64) launch<2>(logits, ld, tokens, experts, top, bias, rule, ids, weights, stream);
    else if(experts <= 128) launch<4>(logits, ld, tokens, experts, top, bias, rule, ids, weights, stream);
    else if(experts <= 256) launch<8>(logits, ld, tokens, experts, top, bias, rule, ids, weights, stream);
    else if(experts <= 512) launch<16>(logits, ld, tokens, experts, top, bias, rule, ids, weights, stream);
    else launch<32>(logits, ld, tokens, experts, top, bias, rule, ids, weights, stream);
    return cudaGetLastError();
}

}  // namespace crit::techniques::topk_routing
