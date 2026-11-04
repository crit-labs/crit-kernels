// Torch entry points over grouped_experts.h: shape checks, the current stream. The expert table is a uint8 CUDA tensor
// holding one Matrix (codes pointer, scales pointer, post) per expert, as the Python layer packs it.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "grouped_experts.h"

namespace gx = crit::techniques::streamed_prefill;

namespace {
void check_ok(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }
void check_cuda(const torch::Tensor& t, const char* name, torch::ScalarType type) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
    TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
}
gx::Routing routing(const torch::Tensor& ids, const torch::Tensor& weights, int64_t experts) {
    check_cuda(ids, "ids", torch::kInt32);
    TORCH_CHECK(ids.dim() == 2, "ids must be [tokens, top_k]");
    gx::Routing r;
    r.ids = ids.data_ptr<std::int32_t>();
    r.tokens = static_cast<int>(ids.size(0));
    r.top_k = static_cast<int>(ids.size(1));
    r.experts = static_cast<int>(experts);
    if(weights.defined() && weights.numel()) {
        check_cuda(weights, "weights", torch::kFloat32);
        TORCH_CHECK(weights.sizes() == ids.sizes(), "weights must be [tokens, top_k]");
        r.weights = weights.data_ptr<float>();
    }
    return r;
}
gx::Workspace workspace(const torch::Tensor& counts, const torch::Tensor& offsets, const torch::Tensor& tile_offsets,
                        const torch::Tensor& row_route, const torch::Tensor& route_row) {
    for(const auto* t : {&counts, &offsets, &tile_offsets, &row_route, &route_row}) check_cuda(*t, "workspace", torch::kInt32);
    return {counts.data_ptr<int>(), offsets.data_ptr<int>(), tile_offsets.data_ptr<int>(), row_route.data_ptr<int>(),
            route_row.data_ptr<int>()};
}
}  // namespace

void route_sort(const torch::Tensor& ids, int64_t experts, torch::Tensor& counts, torch::Tensor& offsets,
                torch::Tensor& tile_offsets, torch::Tensor& row_route, torch::Tensor& route_row) {
    const c10::cuda::CUDAGuard guard(ids.device());
    check_ok(gx::enqueue_route_sort(routing(ids, torch::Tensor(), experts), workspace(counts, offsets, tile_offsets, row_route, route_row),
                                    at::cuda::getCurrentCUDAStream()), "route_sort");
}

void gather(const torch::Tensor& x, const torch::Tensor& ids, int64_t experts, const torch::Tensor& counts,
            const torch::Tensor& offsets, const torch::Tensor& tile_offsets, const torch::Tensor& row_route,
            const torch::Tensor& route_row, torch::Tensor& xp) {
    const c10::cuda::CUDAGuard guard(x.device());
    check_cuda(x, "x", torch::kFloat16);
    check_cuda(xp, "xp", torch::kFloat16);
    check_ok(gx::enqueue_gather(reinterpret_cast<const __half*>(x.data_ptr()), static_cast<int>(x.size(1)),
                                routing(ids, torch::Tensor(), experts), workspace(counts, offsets, tile_offsets, row_route, route_row),
                                reinterpret_cast<__half*>(xp.data_ptr()), at::cuda::getCurrentCUDAStream()), "gather");
}

void grouped_projection(int64_t format, const torch::Tensor& table, const torch::Tensor& xp, int64_t out_dim,
                        const torch::Tensor& ids, int64_t experts, const torch::Tensor& counts, const torch::Tensor& offsets,
                        const torch::Tensor& tile_offsets, const torch::Tensor& row_route, const torch::Tensor& route_row,
                        torch::Tensor& yp) {
    const c10::cuda::CUDAGuard guard(xp.device());
    check_cuda(table, "table", torch::kUInt8);
    TORCH_CHECK(table.numel() == experts * static_cast<int64_t>(sizeof(gx::Matrix)), "table must hold one Matrix per expert");
    check_cuda(xp, "xp", torch::kFloat16);
    check_cuda(yp, "yp", torch::kFloat16);
    check_ok(gx::enqueue_grouped_projection(static_cast<gx::Format>(format), reinterpret_cast<const gx::Matrix*>(table.data_ptr()),
                                            reinterpret_cast<const __half*>(xp.data_ptr()), static_cast<int>(xp.size(1)),
                                            static_cast<int>(out_dim), routing(ids, torch::Tensor(), experts),
                                            workspace(counts, offsets, tile_offsets, row_route, route_row),
                                            reinterpret_cast<__half*>(yp.data_ptr()), at::cuda::getCurrentCUDAStream()),
             "grouped_projection");
}

void swiglu(const torch::Tensor& gate_up, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(gate_up.device());
    check_cuda(gate_up, "gate_up", torch::kFloat16);
    check_cuda(out, "out", torch::kFloat16);
    TORCH_CHECK(gate_up.size(1) == 2 * out.size(1) && gate_up.size(0) == out.size(0), "gate_up must be [rows, 2 * inter]");
    check_ok(gx::enqueue_swiglu(reinterpret_cast<const __half*>(gate_up.data_ptr()), static_cast<int>(out.size(1)),
                                static_cast<int>(out.size(0)), reinterpret_cast<__half*>(out.data_ptr()),
                                at::cuda::getCurrentCUDAStream()), "swiglu");
}

void combine(const torch::Tensor& yp, const torch::Tensor& ids, const torch::Tensor& weights, int64_t experts,
             const torch::Tensor& counts, const torch::Tensor& offsets, const torch::Tensor& tile_offsets,
             const torch::Tensor& row_route, const torch::Tensor& route_row, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(yp.device());
    check_cuda(yp, "yp", torch::kFloat16);
    check_cuda(out, "out", torch::kFloat32);
    check_ok(gx::enqueue_combine(reinterpret_cast<const __half*>(yp.data_ptr()), static_cast<int>(out.size(1)),
                                 routing(ids, weights, experts), workspace(counts, offsets, tile_offsets, row_route, route_row),
                                 out.data_ptr<float>(), at::cuda::getCurrentCUDAStream()), "combine");
}
