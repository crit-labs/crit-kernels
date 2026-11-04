#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("rows(Tensor table, Tensor jobs, int max_rows, Tensor values, Tensor scales, int block, Tensor! out) -> ()");
  ops.def("gemm(Tensor table, Tensor jobs, int max_rows, int max_count, Tensor values, Tensor scales, int block, "
          "Tensor! out) -> ()");
  ops.def("quantize_e2m1x2(Tensor x, float global, Tensor! values, Tensor! scales) -> ()");
  ops.def("quantize_int8(Tensor x, int block, Tensor! values, Tensor! scales) -> ()");
  ops.def("swiglu(Tensor gate_up, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("rows", torch::kCUDA, &rows);
  ops.impl("gemm", torch::kCUDA, &gemm);
  ops.impl("quantize_e2m1x2", torch::kCUDA, &quantize_e2m1x2);
  ops.impl("quantize_int8", torch::kCUDA, &quantize_int8);
  ops.impl("swiglu", torch::kCUDA, &swiglu);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
