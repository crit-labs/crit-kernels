#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("gemv(Tensor x, Tensor codes, Tensor scales, Tensor inverse, bool blocked, Tensor! out) -> ()");
  ops.def("gemv_fp8(Tensor x, Tensor codes, Tensor scales, Tensor! out) -> ()");
  ops.def("gemv_bf16(Tensor x, Tensor w, Tensor! out) -> ()");
  ops.def("hardware_decode(Tensor device_of) -> int");
#if defined(CUDA_KERNEL)
  ops.impl("gemv", torch::kCUDA, &gemv);
  ops.impl("gemv_fp8", torch::kCUDA, &gemv_fp8);
  ops.impl("gemv_bf16", torch::kCUDA, &gemv_bf16);
  ops.impl("hardware_decode", torch::kCUDA, &hardware_decode);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
