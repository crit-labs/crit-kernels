#pragma once

#include <torch/torch.h>

// x [M][K] (M 1..16) BF16 or FP16, codes [N][K / 2] uint8, scales [N][K / 16] E4M3 bytes, inverse [N] FP32 ->
// out [M][N] FP32 or x's dtype.
void gemv(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& scales,
          const torch::Tensor& inverse, torch::Tensor& out);
// x [M][K] (M 1..16) BF16 or FP16, codes [N][K] E4M3 bytes, scales FP32 [N] (one per row) or E8M0 bytes [N][K / 32]
// (MXFP8) -> out [M][N] FP32 or x's dtype.
void gemv_fp8(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& scales, torch::Tensor& out);
// x [M][K] (M 1..16) BF16, w [N][K] BF16 -> out [M][N] FP32 or BF16.
void gemv_bf16(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out);
// 1 when the build's code for the device of `device_of` decodes E2M1 by the hardware conversion, 0 by the bit pattern.
int64_t hardware_decode(const torch::Tensor& device_of);
