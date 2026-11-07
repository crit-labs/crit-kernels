#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("plan(Tensor ids, Tensor? slots, int experts, Tensor! item_slot, Tensor! item_routes, Tensor! item_count) -> ()");
  ops.def("items(Tensor order, Tensor counts, Tensor starts, Tensor item_end, Tensor? slots, int experts, "
          "Tensor! item_slot, Tensor! item_routes, Tensor! item_count) -> ()");
  ops.def("gate_up(Tensor x, int top, Tensor store, int[] layout, bool a4, Tensor item_slot, Tensor item_routes, "
          "Tensor item_count, Tensor! h) -> ()");
  ops.def("down(Tensor h, Tensor store, int[] layout, bool a4, Tensor item_slot, Tensor item_routes, Tensor item_count, "
          "Tensor! y) -> ()");
  ops.def("combine(Tensor y, Tensor ids, Tensor weights, Tensor? slots, int experts, bool skip_negative, Tensor! out) -> ()");
  ops.def("quantize_e2m1x2(Tensor x, float g, Tensor! values, Tensor! scales) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("plan", torch::kCUDA, &plan);
  ops.impl("items", torch::kCUDA, &items);
  ops.impl("gate_up", torch::kCUDA, &gate_up);
  ops.impl("down", torch::kCUDA, &down);
  ops.impl("combine", torch::kCUDA, &combine);
  ops.impl("quantize_e2m1x2", torch::kCUDA, &quantize_e2m1x2);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
