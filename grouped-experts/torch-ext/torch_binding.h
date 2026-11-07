#pragma once

#include <optional>
#include <vector>

#include <torch/torch.h>

// ids [routes] (int64 or int32), slots [experts] int32 or empty; the items int32: item_slot [capacity], item_routes
// [capacity][width] (width 8 or 32), item_count [1]. plan takes up to 1024 routes; items the larger ones from a stable
// sort (order int64 [routes]; counts, starts and item_end int32 [experts]).
void plan(const torch::Tensor& ids, const std::optional<torch::Tensor>& slots, int64_t experts, torch::Tensor& item_slot,
          torch::Tensor& item_routes, torch::Tensor& item_count);
void items(const torch::Tensor& order, const torch::Tensor& counts, const torch::Tensor& starts,
           const torch::Tensor& item_end, const std::optional<torch::Tensor>& slots, int64_t experts,
           torch::Tensor& item_slot, torch::Tensor& item_routes, torch::Tensor& item_count);
// x [tokens][hidden] BF16, store uint8 [slots][bundle bytes] -> h [routes][inter] BF16.
void gate_up(const torch::Tensor& x, int64_t top, const torch::Tensor& store, c10::ArrayRef<int64_t> layout,
             bool a4, const torch::Tensor& item_slot, const torch::Tensor& item_routes, const torch::Tensor& item_count,
             torch::Tensor& h);
// h [routes][inter] BF16 -> y [routes][hidden] FP32, unweighted.
void down(const torch::Tensor& h, const torch::Tensor& store, c10::ArrayRef<int64_t> layout, bool a4,
          const torch::Tensor& item_slot, const torch::Tensor& item_routes, const torch::Tensor& item_count,
          torch::Tensor& y);
// y [routes][hidden] FP32, ids and weights [tokens][top] -> out [tokens][hidden] FP32 in route order.
void combine(const torch::Tensor& y, const torch::Tensor& ids, const torch::Tensor& weights,
             const std::optional<torch::Tensor>& slots, int64_t experts, bool skip_negative, torch::Tensor& out);
// x [rows][cols] FP32 -> values int8 (2 * e2m1), scales FP32 [rows][cols / 16]: crit's declared NVFP4 quantizer.
void quantize_e2m1x2(const torch::Tensor& x, double global, torch::Tensor& values, torch::Tensor& scales);
