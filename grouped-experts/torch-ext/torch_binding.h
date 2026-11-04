#pragma once

#include <torch/torch.h>

// ids [tokens, top_k] int32; the workspace tensors int32: counts [experts], offsets and tile_offsets [experts + 1],
// row_route and route_row [tokens * top_k].
void route_sort(const torch::Tensor& ids, int64_t experts, torch::Tensor& counts, torch::Tensor& offsets,
                torch::Tensor& tile_offsets, torch::Tensor& row_route, torch::Tensor& route_row);
// x [tokens, hidden] fp16 -> xp [tokens * top_k, hidden] fp16 in expert order.
void gather(const torch::Tensor& x, const torch::Tensor& ids, int64_t experts, const torch::Tensor& counts,
            const torch::Tensor& offsets, const torch::Tensor& tile_offsets, const torch::Tensor& row_route,
            const torch::Tensor& route_row, torch::Tensor& xp);
// format 0 bf16, 1 fp8_e4m3_row, 2 nvfp4, 3 mxfp4; table: one Matrix per expert (uint8); xp [rows, in] -> yp [rows, out].
void grouped_projection(int64_t format, const torch::Tensor& table, const torch::Tensor& xp, int64_t out_dim,
                        const torch::Tensor& ids, int64_t experts, const torch::Tensor& counts, const torch::Tensor& offsets,
                        const torch::Tensor& tile_offsets, const torch::Tensor& row_route, const torch::Tensor& route_row,
                        torch::Tensor& yp);
void swiglu(const torch::Tensor& gate_up, torch::Tensor& out);
// yp [rows, hidden] fp16 -> out [tokens, hidden] fp32, each token's routes in order with their weights.
void combine(const torch::Tensor& yp, const torch::Tensor& ids, const torch::Tensor& weights, int64_t experts,
             const torch::Tensor& counts, const torch::Tensor& offsets, const torch::Tensor& tile_offsets,
             const torch::Tensor& row_route, const torch::Tensor& route_row, torch::Tensor& out);
