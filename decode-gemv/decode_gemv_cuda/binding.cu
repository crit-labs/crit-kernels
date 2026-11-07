// Torch entry points over decode_gemv.h: shape, dtype and alignment checks, the current stream. Nothing allocates and
// nothing waits for the device, so a CUDA graph captures the calls.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "decode_gemv.h"

namespace nr = crit::kernels::decode_gemv;

namespace {
void check_ok(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }

bool aligned16(const torch::Tensor& t) { return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16 == 0; }

void check_rows(const torch::Tensor& x, const char* what) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2, what, ": x is a 2-D CUDA tensor [rows][K]");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= nr::kMaxRows, what, ": 1 to ", nr::kMaxRows, " rows, not ",
                x.size(0));
    TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && aligned16(x), what,
                ": x's rows are contiguous and start on 16-byte boundaries");
}

void check_out(const torch::Tensor& out, const torch::Tensor& x, int64_t n, const char* what) {
    TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.size(0) == x.size(0) && out.size(1) == n && out.stride(1) == 1,
                what, ": out is [rows][N] with contiguous rows");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32 || out.scalar_type() == x.scalar_type(), what,
                ": out is FP32 or x's dtype");
}
}  // namespace

void gemv(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& scales,
          const torch::Tensor& inverse, torch::Tensor& out) {
    check_rows(x, "gemv");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat16, "gemv: x is BF16 or FP16");
    const int64_t n = codes.size(0), k = x.size(1);
    TORCH_CHECK(k % 32 == 0, "gemv: K is a multiple of 32, not ", k);
    TORCH_CHECK(codes.is_cuda() && codes.is_contiguous() && codes.scalar_type() == torch::kUInt8 && codes.dim() == 2 &&
                    codes.size(1) * 2 == k && aligned16(codes),
                "gemv: codes are contiguous uint8 [N][K / 2] on 16 bytes");
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.element_size() == 1 && scales.dim() == 2 &&
                    scales.size(0) == n && scales.size(1) * 16 == k,
                "gemv: scales are contiguous E4M3 bytes [N][K / 16]");
    TORCH_CHECK(inverse.is_cuda() && inverse.is_contiguous() && inverse.scalar_type() == torch::kFloat32 &&
                    inverse.numel() == n,
                "gemv: inverse is contiguous FP32 [N]");
    check_out(out, x, n, "gemv");
    const c10::cuda::CUDAGuard guard(x.device());
    check_ok(nr::enqueue_gemv(x.data_ptr(), static_cast<int>(x.stride(0)), x.scalar_type() == torch::kBFloat16,
                              codes.data_ptr<std::uint8_t>(), reinterpret_cast<const std::uint8_t*>(scales.data_ptr()),
                              inverse.data_ptr<float>(), out.data_ptr(), static_cast<int>(out.stride(0)),
                              out.scalar_type() == torch::kFloat32, static_cast<int>(x.size(0)), static_cast<int>(n),
                              static_cast<int>(k), at::cuda::getCurrentCUDAStream()),
             "gemv");
}

void gemv_fp8(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& scales, torch::Tensor& out) {
    check_rows(x, "gemv_fp8");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat16,
                "gemv_fp8: x is BF16 or FP16");
    const int64_t n = codes.size(0), k = x.size(1);
    TORCH_CHECK(codes.is_cuda() && codes.is_contiguous() && codes.element_size() == 1 && codes.dim() == 2 &&
                    codes.size(1) == k && k % 16 == 0 && aligned16(codes),
                "gemv_fp8: codes are contiguous E4M3 bytes [N][K] on 16 bytes, K a multiple of 16");
    const bool mx = scales.scalar_type() == torch::kUInt8;
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous(), "gemv_fp8: scales are a contiguous CUDA tensor");
    if (mx) {
        TORCH_CHECK(scales.dim() == 2 && scales.size(0) == n && k % 32 == 0 && scales.size(1) * 32 == k,
                    "gemv_fp8: MXFP8 scales are E8M0 bytes [N][K / 32], K a multiple of 32");
    } else {
        TORCH_CHECK(scales.scalar_type() == torch::kFloat32 && scales.numel() == n,
                    "gemv_fp8: row scales are FP32 [N] (or E8M0 bytes [N][K / 32] for MXFP8)");
    }
    check_out(out, x, n, "gemv_fp8");
    const c10::cuda::CUDAGuard guard(x.device());
    check_ok(nr::enqueue_gemv_fp8(x.data_ptr(), static_cast<int>(x.stride(0)), x.scalar_type() == torch::kBFloat16,
                                  reinterpret_cast<const std::uint8_t*>(codes.data_ptr()),
                                  mx ? nullptr : scales.data_ptr<float>(),
                                  mx ? scales.data_ptr<std::uint8_t>() : nullptr, out.data_ptr(),
                                  static_cast<int>(out.stride(0)), out.scalar_type() == torch::kFloat32,
                                  static_cast<int>(x.size(0)), static_cast<int>(n), static_cast<int>(k),
                                  at::cuda::getCurrentCUDAStream()),
             "gemv_fp8");
}

void gemv_bf16(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out) {
    check_rows(x, "gemv_bf16");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "gemv_bf16: x is BF16");
    const int64_t n = w.size(0), k = x.size(1);
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kBFloat16 && w.dim() == 2 &&
                    w.size(1) == k && k % 8 == 0 && aligned16(w),
                "gemv_bf16: w is contiguous BF16 [N][K] on 16 bytes, K a multiple of 8");
    check_out(out, x, n, "gemv_bf16");
    const c10::cuda::CUDAGuard guard(x.device());
    check_ok(nr::enqueue_gemv_bf16(x.data_ptr(), static_cast<int>(x.stride(0)), w.data_ptr(), out.data_ptr(),
                                   static_cast<int>(out.stride(0)), out.scalar_type() == torch::kFloat32,
                                   static_cast<int>(x.size(0)), static_cast<int>(n), static_cast<int>(k),
                                   at::cuda::getCurrentCUDAStream()),
             "gemv_bf16");
}

int64_t hardware_decode(const torch::Tensor& device_of) {
    const c10::cuda::CUDAGuard guard(device_of.device());
    return nr::hardware_decode();
}
