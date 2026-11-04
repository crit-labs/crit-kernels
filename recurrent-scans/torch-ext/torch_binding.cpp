#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("causal_conv1d(Tensor! out, Tensor! state, Tensor x, Tensor weight, Tensor? bias, bool silu) -> ()");
  ops.def("gated_delta_prefill(Tensor! out, Tensor! state, Tensor q, Tensor k, Tensor v, Tensor log_decay, "
          "Tensor beta, float scale) -> ()");
  ops.def("gated_delta_scan(Tensor! out, Tensor! state, Tensor q, Tensor k, Tensor v, Tensor log_decay, "
          "Tensor beta, float scale, Tensor? checkpoints) -> ()");
  ops.def("ssd_prefill(Tensor! out, Tensor! state, Tensor x, Tensor dt, Tensor B, Tensor C, Tensor A, Tensor? D) -> ()");
  ops.def("ssd_scan(Tensor! out, Tensor! state, Tensor x, Tensor dt, Tensor B, Tensor C, Tensor A, Tensor? D, "
          "Tensor? checkpoints) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("causal_conv1d", torch::kCUDA, &causal_conv1d);
  ops.impl("gated_delta_prefill", torch::kCUDA, &gated_delta_prefill);
  ops.impl("gated_delta_scan", torch::kCUDA, &gated_delta_scan);
  ops.impl("ssd_prefill", torch::kCUDA, &ssd_prefill);
  ops.impl("ssd_scan", torch::kCUDA, &ssd_scan);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
