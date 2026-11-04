// Grouped expert projections for prompt chunks, for any block weight format: the chunk's routes are sorted by expert
// (stable, so the order depends on the routes alone), the routed rows gathered, every expert's rows multiplied by its
// weights in one launch (FP16 tensor-core tiles, weights decoded tile by tile through a format hook), and each
// token's expert outputs combined in route order with its routing weights. No atomics and no reduction split by the
// row count: a token's result does not depend on the other tokens of its chunk, so chunking can follow an expert
// streaming schedule (crit's expert-streamed-prefill technique) freely.
//
// A baseline to replace, not the answer: it runs any of these formats on any NVIDIA GPU from sm_70 so a schedule works
// before the engine has chosen its GEMM. A provider's grouped GEMM on the format's own tensor cores (CUTLASS's
// ptr-array grouped GEMM, block-scaled on sm_100 and sm_120) is usually faster: measure both on the engine's shapes.
//
// The weights of an expert come from a pointer table, so each expert may sit wherever the engine keeps it: a device
// cache slot, a staging buffer the schedule filled over the link, or device-visible host memory.
//
// CUDA only (nvcuda::wmma, sm_70 and newer). A format's native low-precision tensor cores (FP8, FP4) can be faster
// than this kernel's FP16 tiles; a provider's grouped GEMM for that format is then the yardstick and the fast path.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace crit::techniques::streamed_prefill {

// Stored weight formats the tile decoder reads. Every matrix is [rows][cols] (output features by input features),
// codes row-major.
enum class Format : int {
    bf16 = 0,          // 2 bytes an element; scales unused
    fp8_e4m3_row = 1,  // 1 byte an element; scales: one float per row
    nvfp4 = 2,         // E2M1, two a byte (low nibble the even element); scales: E4M3 per 16 elements, [rows][cols / 16]
    mxfp4 = 3,         // E2M1 as nvfp4; scales: E8M0 per 32 elements, [rows][cols / 32]
};

// One expert's matrix as stored. A weight is code * block scale * post (nvfp4: post is the checkpoint's 1 / global
// scale; 1 for the others). Row scales (fp8_e4m3_row) and post are applied to the FP32 accumulator, so nvfp4 and
// fp8 tiles decode to FP16 exactly.
struct Matrix {
    const std::uint8_t* codes = nullptr;
    const std::uint8_t* scales = nullptr;
    float post = 1.f;
};

// A chunk's routing: ids[t * top_k + j] (an id outside [0, experts) is skipped) and their weights.
struct Routing {
    const std::int32_t* ids = nullptr;
    const float* weights = nullptr;
    int tokens = 0, top_k = 0, experts = 0;
};

// Device workspace for one chunk of at most `tokens * top_k` routed rows.
struct Workspace {
    int* counts = nullptr;        // [experts]
    int* offsets = nullptr;       // [experts + 1]: expert e's rows are [offsets[e], offsets[e + 1])
    int* tile_offsets = nullptr;  // [experts + 1]: expert e's row tiles start at tile_offsets[e]
    int* row_route = nullptr;     // [tokens * top_k]: the route (t * top_k + j) of each sorted row
    int* route_row = nullptr;     // [tokens * top_k]: the sorted row of each route, -1 for a skipped id
};

constexpr int kTileRows = 64, kTileCols = 64, kTileDepth = 32;

// Sorts the routes by expert, routes in increasing order within an expert, and fills the workspace.
cudaError_t enqueue_route_sort(const Routing& r, const Workspace& w, cudaStream_t stream);
// xp[row] = x[token of row_route[row]], `hidden` FP16 values a row; rows = tokens * top_k (skipped routes leave
// trailing rows unused).
cudaError_t enqueue_gather(const __half* x, int hidden, const Routing& r, const Workspace& w, __half* xp,
                           cudaStream_t stream);
// yp[row][0, out_dim) = xp[row][0, in_dim) * W_e^T for each expert e's rows, W_e = experts[e] ([out_dim][in_dim],
// a device array of Matrix). out_dim a multiple of 64, in_dim a multiple of 32.
cudaError_t enqueue_grouped_projection(Format format, const Matrix* experts, const __half* xp, int in_dim, int out_dim,
                                       const Routing& r, const Workspace& w, __half* yp, cudaStream_t stream);
// out[row][i] = silu(gate_up[row][i]) * gate_up[row][inter + i]: the gate rows first in a fused gate/up matrix.
cudaError_t enqueue_swiglu(const __half* gate_up, int inter, int rows, __half* out, cudaStream_t stream);
// out[t][h] = sum over j in route order of weights[t * top_k + j] * yp[route_row[t * top_k + j]][h], in FP32.
cudaError_t enqueue_combine(const __half* yp, int hidden, const Routing& r, const Workspace& w, float* out,
                            cudaStream_t stream);

} // namespace crit::techniques::streamed_prefill
