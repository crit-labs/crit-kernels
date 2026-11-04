#pragma once

#include <torch/torch.h>

// table: one Matrix per matrix (uint8); jobs: RowJob or GemmJob records (uint8); codes [rows, cols] uint8 E4M3 and
// scales [rows, cols / group] fp32: the quantized activations; out [rows, out_ld] fp32.
void rows(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, const torch::Tensor& codes,
          const torch::Tensor& scales, int64_t group, torch::Tensor& out);
void gemm(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, int64_t max_count,
          const torch::Tensor& codes, const torch::Tensor& scales, int64_t group, torch::Tensor& out);
// x [rows, cols] fp32 or bf16 -> codes uint8 E4M3 and scales fp32 per group (kind 0: MXFP8, 1: amax / 448).
void quantize(const torch::Tensor& x, int64_t group, int64_t kind, torch::Tensor& codes, torch::Tensor& scales);
