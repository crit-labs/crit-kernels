// Matched FP8 projections: E4M3 weights (MXFP8's power-of-two block scales, an FP8 checkpoint's FP32 128 x 128 block
// scales, or one scale per row) against E4M3 activations quantized per group, in crit's declared per-output order
// (crit inference/quantized-projections, fp8_block_device.cuh): per 32-element block, the block's products summed by
// one e4m3 mma.sync m16n8k32 with a zero accumulator (weights the A operand, tokens B, crit's fragment order); lane
// b % 16 = fma(block sum, weight scale * activation scale, lane) with the product rounded, blocks in order; the 16 lanes
// reduced by the halving tree; times the matrix's post.
//
// Two kernels give the same bits for every output: `rows` for decode (16 rows a CTA, the lanes split over four warps,
// 1 to 8 tokens a job) and `gemm` for prompt chunks (a warp 16 rows x 32 tokens, the lanes folded as they finish). An
// output depends only on its row and its token, not on the job's other tokens or the kernel, so decode, verify rounds,
// prefill and a prompt cache agree bit for bit. The quantizer is crit's declared one (MXFP8's shared exponent or amax /
// 448, E4M3 round to nearest even with saturation). NVIDIA sm_89 and newer (e4m3 mma.sync).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace crit::techniques::fp8_block {

// One matrix: codes [rows][cols] E4M3; scales [ceil(rows / scale_rows)][cols / scale_k], FP32 or (scale_e8m0) MXFP8's
// E8M0 bytes as stored (MXFP8: 1, 32; 128 x 128 blocks: 128, 128; per row: 1, cols). cols and scale_k multiples of 32.
// An E8M0 byte e is the scale 2^(e - 127), the same value as its FP32 decoding, so both give the same bits.
struct Matrix {
    const std::uint8_t* codes = nullptr;
    const void* scales = nullptr;
    int rows = 0, cols = 0, scale_rows = 1, scale_k = 32;
    float post = 1.0f;
    int scale_e8m0 = 0;
};

// Quantized activations: codes [row][ld] E4M3, scales [row][lds] FP32, one per `group` codes (a multiple of 32).
struct Acts {
    const std::uint8_t* codes = nullptr;
    const float* scales = nullptr;
    int ld = 0, lds = 0, group = 32;
};

enum class ScaleKind : int { mx = 0, amax = 1 };

constexpr int kMaxJobTokens = 8;

// Decode: one matrix and up to 8 activation rows; out[out_row[j]][r] for its tokens j and rows r < rows.
struct RowJob {
    int matrix = 0;
    int tokens = 0;
    int act_row[kMaxJobTokens] = {};
    int out_row[kMaxJobTokens] = {};
};

// Prompt: one matrix and `count` consecutive activation rows from `first`; out[first + i][r].
struct GemmJob {
    int matrix = 0;
    int first = 0, count = 0;
};

// matrices: a device array of Matrix; jobs: a device array; out [..][out_ld] FP32.
cudaError_t enqueue_rows(const Matrix* matrices, const RowJob* jobs, int job_count, int max_rows, const Acts& acts,
                         float* out, int out_ld, cudaStream_t stream);
cudaError_t enqueue_gemm(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows, int max_count,
                         const Acts& acts, float* out, int out_ld, cudaStream_t stream);

// The declared quantizer: rows x cols values (x [row][ld_x], FP32 or BF16) into acts, one scale per acts.group.
cudaError_t enqueue_quantize(const void* x, bool bf16, int ld_x, int rows, int cols, ScaleKind kind, const Acts& acts,
                             cudaStream_t stream);

}  // namespace crit::techniques::fp8_block
