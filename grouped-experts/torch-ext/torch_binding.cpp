#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("route_sort(Tensor ids, int experts, Tensor! counts, Tensor! offsets, Tensor! tile_offsets, Tensor! row_route, "
          "Tensor! route_row) -> ()");
  ops.def("gather(Tensor x, Tensor ids, int experts, Tensor counts, Tensor offsets, Tensor tile_offsets, Tensor row_route, "
          "Tensor route_row, Tensor! xp) -> ()");
  ops.def("grouped_projection(int format, Tensor table, Tensor xp, int out_dim, Tensor ids, int experts, Tensor counts, "
          "Tensor offsets, Tensor tile_offsets, Tensor row_route, Tensor route_row, Tensor! yp) -> ()");
  ops.def("swiglu(Tensor gate_up, Tensor! out) -> ()");
  ops.def("combine(Tensor yp, Tensor ids, Tensor weights, int experts, Tensor counts, Tensor offsets, Tensor tile_offsets, "
          "Tensor row_route, Tensor route_row, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("route_sort", torch::kCUDA, &route_sort);
  ops.impl("gather", torch::kCUDA, &gather);
  ops.impl("grouped_projection", torch::kCUDA, &grouped_projection);
  ops.impl("swiglu", torch::kCUDA, &swiglu);
  ops.impl("combine", torch::kCUDA, &combine);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
