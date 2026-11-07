#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "mm_rows(Tensor x, Tensor w, Tensor? scales, Tensor? row_scale, int weights, int acts, float input_scale, "
      "bool blocked, int groups, int warps, int splits, Tensor(c!)? scratch, Tensor(a!)? partial, Tensor(b!)? counters, "
      "Tensor! out) -> ()");
  ops.def("quantize_fp8_rows(Tensor x, Tensor! q, Tensor! scale) -> ()");
  ops.def("quantize_nvfp4(Tensor x, float input_scale, Tensor! codes, Tensor! scales) -> ()");
  ops.def("step_values(int weights) -> int");
#if defined(CUDA_KERNEL)
  ops.impl("mm_rows", torch::kCUDA, &mm_rows);
  ops.impl("quantize_fp8_rows", torch::kCUDA, &quantize_fp8_rows);
  ops.impl("quantize_nvfp4", torch::kCUDA, &quantize_nvfp4);
  ops.impl("step_values", &step_values);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
