// Torch entry points over int_block.h: shape checks and the current stream. The matrix table is a uint8 CUDA tensor
// holding one Matrix per matrix and the job lists uint8 CUDA tensors of RowJob or GemmJob, as the Python layer packs
// them.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "int_block.h"

namespace ib = crit::techniques::int_block;

namespace {
void check_ok(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }
void check_cuda(const torch::Tensor& t, const char* name, torch::ScalarType type) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
    TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
}
ib::Acts acts_of(const torch::Tensor& values, const torch::Tensor& scales, int64_t block) {
    check_cuda(values, "values", torch::kInt8);
    check_cuda(scales, "scales", torch::kFloat32);
    TORCH_CHECK(values.dim() == 2 && scales.dim() == 2 && values.size(0) == scales.size(0), "values [rows, cols] and "
                "scales [rows, cols / block]");
    return {values.data_ptr<std::int8_t>(), scales.data_ptr<float>(), static_cast<int>(values.size(1)),
            static_cast<int>(scales.size(1)), static_cast<int>(block)};
}
}  // namespace

void rows(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, const torch::Tensor& values,
          const torch::Tensor& scales, int64_t block, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(values.device());
    check_cuda(table, "table", torch::kUInt8);
    check_cuda(jobs, "jobs", torch::kUInt8);
    check_cuda(out, "out", torch::kFloat32);
    TORCH_CHECK(jobs.numel() % static_cast<int64_t>(sizeof(ib::RowJob)) == 0, "jobs must hold RowJob records");
    check_ok(ib::enqueue_rows(reinterpret_cast<const ib::Matrix*>(table.data_ptr()),
                              reinterpret_cast<const ib::RowJob*>(jobs.data_ptr()),
                              static_cast<int>(jobs.numel() / static_cast<int64_t>(sizeof(ib::RowJob))),
                              static_cast<int>(max_rows), acts_of(values, scales, block), out.data_ptr<float>(),
                              static_cast<int>(out.size(1)), at::cuda::getCurrentCUDAStream()), "rows");
}

void gemm(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, int64_t max_count,
          const torch::Tensor& values, const torch::Tensor& scales, int64_t block, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(values.device());
    check_cuda(table, "table", torch::kUInt8);
    check_cuda(jobs, "jobs", torch::kUInt8);
    check_cuda(out, "out", torch::kFloat32);
    TORCH_CHECK(jobs.numel() % static_cast<int64_t>(sizeof(ib::GemmJob)) == 0, "jobs must hold GemmJob records");
    check_ok(ib::enqueue_gemm(reinterpret_cast<const ib::Matrix*>(table.data_ptr()),
                              reinterpret_cast<const ib::GemmJob*>(jobs.data_ptr()),
                              static_cast<int>(jobs.numel() / static_cast<int64_t>(sizeof(ib::GemmJob))),
                              static_cast<int>(max_rows), static_cast<int>(max_count), acts_of(values, scales, block),
                              out.data_ptr<float>(), static_cast<int>(out.size(1)), at::cuda::getCurrentCUDAStream()),
             "gemm");
}

void quantize_e2m1x2(const torch::Tensor& x, double global, torch::Tensor& values, torch::Tensor& scales) {
    const c10::cuda::CUDAGuard guard(x.device());
    check_cuda(x, "x", torch::kFloat32);
    check_ok(ib::enqueue_quantize_e2m1x2(x.data_ptr<float>(), static_cast<int>(x.size(1)), static_cast<int>(x.size(0)),
                                         static_cast<int>(x.size(1)), static_cast<float>(global),
                                         acts_of(values, scales, 16), at::cuda::getCurrentCUDAStream()), "quantize_e2m1x2");
}

void quantize_int8(const torch::Tensor& x, int64_t block, torch::Tensor& values, torch::Tensor& scales) {
    const c10::cuda::CUDAGuard guard(x.device());
    check_cuda(x, "x", torch::kFloat32);
    check_ok(ib::enqueue_quantize_int8(x.data_ptr<float>(), static_cast<int>(x.size(1)), static_cast<int>(x.size(0)),
                                       static_cast<int>(x.size(1)), acts_of(values, scales, block),
                                       at::cuda::getCurrentCUDAStream()), "quantize_int8");
}

void swiglu(const torch::Tensor& gate_up, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(gate_up.device());
    check_cuda(gate_up, "gate_up", torch::kFloat32);
    check_cuda(out, "out", torch::kFloat32);
    TORCH_CHECK(gate_up.size(1) == 2 * out.size(1) && gate_up.size(0) == out.size(0), "gate_up must be [rows, 2 * inter]");
    check_ok(ib::enqueue_swiglu(gate_up.data_ptr<float>(), static_cast<int>(out.size(1)), static_cast<int>(out.size(0)),
                                out.data_ptr<float>(), at::cuda::getCurrentCUDAStream()), "swiglu");
}
