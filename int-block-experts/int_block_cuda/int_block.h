// Integer block expert projections: 4-bit weights (NVFP4's E2M1 codes or INT4, one E4M3 scale per 16) against
// activations quantized per block (E2M1 x 2 for 4-bit activations, INT8 for 8-bit ones), in crit's declared integer
// block order (crit inference/host-expert-arithmetic, host_int.h): each 16-block's dot of the integer values is exact
// (dp4a, int8 tensor cores), block b goes to lane b % 16, lane = fma(float(sum), weight scale * activation scale, lane)
// with the product rounded, blocks in order, the 16 lanes reduced by the halving tree, times the matrix's post.
//
// Two kernels give the same bits for every output: `rows` for decode (a half-warp per output row, every token of a
// job reading the row once) and `gemm` for prompt chunks (int8 m16n8k16 tensor-core tiles, the FP32 step per block in
// registers). crit's CPU rows (host_int.h, AVX-512 VNNI and AVX2) give them too, so an expert's output does not depend
// on the processor or the kernel that computed it: a CPU tier, a device cache and the prompt path agree bit for bit.
//
// The quantizers, SiLU and SwiGLU are crit's declared ones (IEEE operations only), so an expert's intermediate is the
// same codes wherever it is computed. CUDA, sm_80 and newer (int8 mma.sync m16n8k16; dp4a).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace crit::techniques::int_block {

// One matrix as stored, rows layout: codes [rows][cols / 2] (low nibble the even element), E4M3 block scales
// [rows][cols / 16]; a weight is value * scale * post (value E2M1 x 2 or INT4). cols a multiple of 64.
struct Matrix {
    const std::uint8_t* codes = nullptr;
    const std::uint8_t* scales = nullptr;
    int rows = 0, cols = 0;
    float post = 1.0f;
    int int4 = 0;   // 1: INT4 values; 0: E2M1 x 2
};

// Quantized activations: values [row][ld] int8, scales [row][lds] FP32, one per `block` (16 or 32) values.
struct Acts {
    const std::int8_t* values = nullptr;
    const float* scales = nullptr;
    int ld = 0, lds = 0, block = 16;
};

constexpr int kMaxJobTokens = 8;

// Decode: a job is one matrix and up to 8 activation rows; out[out_row[j]][r] for its tokens j and rows r < rows.
struct RowJob {
    int matrix = 0;
    int tokens = 0;
    int act_row[kMaxJobTokens] = {};
    int out_row[kMaxJobTokens] = {};
};

// Prompt: a job is one matrix and `count` consecutive activation rows from `first`; out[first + i][r].
struct GemmJob {
    int matrix = 0;
    int first = 0, count = 0;
};

// matrices: a device array of Matrix; jobs: a device array; out [..][out_ld] FP32.
cudaError_t enqueue_rows(const Matrix* matrices, const RowJob* jobs, int job_count, int max_rows, const Acts& acts,
                         float* out, int out_ld, cudaStream_t stream);
cudaError_t enqueue_gemm(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows, int max_count,
                         const Acts& acts, float* out, int out_ld, cudaStream_t stream);

// Declared quantizers of rows x cols FP32 values (x [row][ld_x]) into acts (values [row][acts.ld], scales
// [row][acts.lds]): E2M1 x 2 per 16 with crit's NVFP4 global divisor, or INT8 per acts.block.
cudaError_t enqueue_quantize_e2m1x2(const float* x, int ld_x, int rows, int cols, float global, const Acts& acts,
                                    cudaStream_t stream);
cudaError_t enqueue_quantize_int8(const float* x, int ld_x, int rows, int cols, const Acts& acts, cudaStream_t stream);
// out[row][i] = silu(gate_up[row][i]) * gate_up[row][inter + i], declared (crit's swiglu_declared).
cudaError_t enqueue_swiglu(const float* gate_up, int inter, int rows, float* out, cudaStream_t stream);

}  // namespace crit::techniques::int_block
