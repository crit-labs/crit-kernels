"""decode-gemv against an FP64 reference of the stored values: the CUDA and the Triton GEMV within FP32 accumulation,
each row's bits the same alone and beside up to 7 others, both E2M1 decodes the same bits, the decoders one FP32
product rounded once, and the CUDA GEMVs captured in a CUDA graph."""
import os
import shutil
import sys

import pytest
import torch

import kernels

nr = kernels.get_kernel("crit-labs/decode-gemv", version=1)
dev = torch.device("cuda")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    dtype=torch.float64)


def matrix(n, k, seed=5):
    """Random codes, E4M3 block scales and inverse global scales [N]."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8)
    scales = (torch.rand(n, k // 16, generator=g) * 4 + 0.01).to(torch.float8_e4m3fn)
    inverse = (torch.rand(n, generator=g) + 0.5) / 64
    return codes.to(dev), scales.to(dev), inverse.to(dev)


def values(codes, scales, inverse):
    """The stored values in FP64: e2m1 * e4m3 * inverse [N][K]."""
    c = codes.cpu().long()
    v = torch.stack((E2M1[c & 15], E2M1[c >> 4]), dim=-1).reshape(c.shape[0], -1)
    v = v.view(c.shape[0], -1, 16) * scales.cpu().double()[..., None]
    return v.reshape(c.shape[0], -1) * inverse.cpu().double()[:, None]


def activations(m, k, dtype=torch.bfloat16, seed=8):
    return torch.randn(m, k, generator=torch.Generator().manual_seed(seed)).to(dev, dtype)


def nvfp4_gemv(x, c, s, i, dtype=torch.float32, path="cuda"):
    out = torch.empty(x.shape[0], c.shape[0], dtype=dtype, device=dev)
    (nr.gemv if path == "cuda" else nr.gemv_triton)(x, c, s, i, out)
    return out


@pytest.mark.kernels_ci
@pytest.mark.parametrize("path", ["cuda", "triton"])
@pytest.mark.parametrize("shape", [(96, 256), (1000, 5120), (130, 2048)])
def test_the_gemv_multiplies_by_the_stored_values_and_keeps_rows(path, shape):
    n, k = shape
    c, s, i = matrix(n, k)
    most = nr.CUDA_ROWS if path == "cuda" else nr.TRITON_ROWS
    x = activations(most, k)
    exact = x.double() @ values(c, s, i).t().to(dev)
    batch = nvfp4_gemv(x, c, s, i, path=path)
    magnitude = (x.double().abs() @ values(c, s, i).abs().t().to(dev))
    assert bool(((batch.double() - exact).abs() <= magnitude * 2e-5 + 1e-30).all())
    for rows in range(1, most + 1):
        part = nvfp4_gemv(x[:rows], c, s, i, path=path)
        for r in range(rows):
            assert torch.equal(part[r], nvfp4_gemv(x[r:r + 1].clone(), c, s, i, path=path)[0]), (rows, r)
    assert torch.equal(nvfp4_gemv(x, c, s, i, path=path), batch)   # a repeat


@pytest.mark.kernels_ci
def test_16_bit_outputs_and_fp16_activations():
    c, s, i = matrix(96, 256)
    x = activations(3, 256)
    full = nvfp4_gemv(x, c, s, i)
    assert torch.equal(nvfp4_gemv(x, c, s, i, torch.bfloat16), full.to(torch.bfloat16))
    half = nvfp4_gemv(x.half(), c, s, i, torch.float16)
    assert half.dtype == torch.float16
    exact = x.half().double() @ values(c, s, i).t().to(dev)
    assert float((half.double() - exact).abs().max() / exact.abs().max()) < 1e-3


@pytest.mark.kernels_ci
def test_the_cuda_gemv_refuses_what_it_cannot_read():
    c, s, i = matrix(96, 256)
    x = activations(9, 256)
    with pytest.raises(RuntimeError):
        nvfp4_gemv(activations(17, 256), c, s, i)                  # 17 rows
    with pytest.raises(RuntimeError):
        nvfp4_gemv(activations(2, 260)[:, 4:], c, s, i)          # rows not on 16-byte boundaries
    with pytest.raises(RuntimeError):
        nvfp4_gemv(x[:2].float(), c, s, i)                        # FP32 activations


@pytest.mark.kernels_ci
def test_the_bf16_gemv_multiplies_and_keeps_rows():
    g = torch.Generator().manual_seed(3)
    w = (torch.randn(1000, 2048, generator=g) * 0.05).to(dev, torch.bfloat16)
    x = activations(16, 2048)
    exact = x.double() @ w.double().t()
    out = torch.empty(16, 1000, device=dev)
    nr.gemv_bf16(x, w, out)
    assert float((out.double() - exact).abs().max() / exact.abs().max()) < 1e-5
    for rows in range(1, 17):
        part = torch.empty(rows, 1000, device=dev)
        nr.gemv_bf16(x[:rows], w, part)
        assert torch.equal(part, out[:rows]), rows


@pytest.mark.kernels_ci
def test_the_triton_gemv_gives_the_same_bits_with_either_e2m1_decode():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("the hardware E2M1 conversion needs compute capability 10 or later")
    switch = sys.modules[nr.gemv_triton.__module__]._HW
    c, s, i = matrix(96, 256)
    x = activations(8, 256)
    found = {}
    try:
        for hardware in (True, False):
            switch[dev] = hardware
            found[hardware] = [nvfp4_gemv(x[:rows], c, s, i, path="triton") for rows in (1, 3, 8)]
    finally:
        switch.pop(dev, None)
    for a, b in zip(found[True], found[False]):
        assert torch.equal(a, b)


def test_cuda_and_triton_agree_within_fp32_rounding():
    c, s, i = matrix(1000, 5120)
    x = activations(8, 5120)
    a, b = nvfp4_gemv(x, c, s, i), nvfp4_gemv(x, c, s, i, path="triton")
    assert float((a - b).abs().max() / b.abs().max()) < 1e-5


@pytest.mark.kernels_ci
def test_the_decoders_round_one_fp32_product_once():
    c, s, i = matrix(96, 256)
    product = (values(c, s, torch.ones_like(i)).float().to(dev) * i[:, None])   # FP32: e2m1 * scale exact, * inverse
    for dtype in (torch.bfloat16, torch.float16):
        out = torch.empty(96, 256, dtype=dtype, device=dev)
        assert torch.equal(nr.decode(c, s, i, out), product.to(dtype)), dtype
    factor = i * 64
    out = torch.empty(96, 256, dtype=torch.float8_e4m3fn, device=dev)
    want = (values(c, s, torch.ones_like(i)).float().to(dev) * factor[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(nr.decode_fp8(c, s, factor, out).view(torch.uint8), want.view(torch.uint8))


@pytest.mark.kernels_ci
def test_the_gemvs_capture_in_a_cuda_graph():
    c, s, i = matrix(1000, 2048)
    w = (torch.randn(1000, 2048, generator=torch.Generator().manual_seed(1)) * 0.05).to(dev, torch.bfloat16)
    x = activations(4, 2048)
    out, out16 = torch.empty(4, 1000, device=dev), torch.empty(4, 1000, device=dev)
    eager = nvfp4_gemv(x, c, s, i)
    nr.gemv_bf16(x, w, out16)
    eager16 = out16.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        nr.gemv(x, c, s, i, out)
        nr.gemv_bf16(x, w, out16)
    out.zero_()
    out16.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager) and torch.equal(out16, eager16)


@pytest.mark.kernels_ci
def test_the_cuda_gemv_gives_the_same_bits_with_either_e2m1_decode():
    """The build's code decodes by the hardware conversion on sm_100a and later; a build of the same source for the
    plain target (sm_120 for sm_120a) decodes by the bit pattern: both must give the same bits. Builds that second
    variant with torch's extension builder (skipped without nvcc)."""
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip("the hardware E2M1 conversion needs compute capability 10 or later")
    if not nr.hardware_decode():
        pytest.skip("this build decodes by the bit pattern already")
    if shutil.which("nvcc") is None and not os.environ.get("CUDA_HOME"):
        pytest.skip("no CUDA compiler for the bit-pattern variant")
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    before = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        plain = load(name=f"decode_gemv_plain_{major}{minor}",
                     sources=[os.path.join(here, "tests", "plain_variant.cu"),
                              os.path.join(here, "decode_gemv_cuda", "decode_gemv.cu")],
                     extra_include_paths=[os.path.join(here, "decode_gemv_cuda")], extra_cuda_cflags=["-O3"])
    finally:
        if before is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = before
    assert plain.hardware_decode() == 0
    c, s, i = matrix(96, 256)
    x = activations(8, 256)
    for rows in (1, 3, 8):
        out = torch.empty(rows, 96, device=dev)
        plain.gemv(x[:rows], c, s, i, out)
        assert torch.equal(out, nvfp4_gemv(x[:rows], c, s, i)), rows


def fp8_matrix(n, k, mx, seed=6):
    """E4M3 codes (no NaN) and FP32 row scales, or E8M0 exponents per 32 inputs (MXFP8)."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 0x7E, (n, k), generator=g, dtype=torch.uint8)
    codes = codes | (torch.randint(0, 2, (n, k), generator=g, dtype=torch.uint8) << 7)
    if mx:
        scales = torch.randint(118, 132, (n, k // 32), generator=g, dtype=torch.uint8)
    else:
        scales = torch.rand(n, generator=g) / 100 + 1e-3
    return codes.to(dev), scales.to(dev)


def fp8_values(codes, scales):
    v = codes.cpu().view(torch.float8_e4m3fn).double()
    if scales.dtype == torch.uint8:
        return v * torch.exp2(scales.cpu().double() - 127).repeat_interleave(32, dim=1)
    return v * scales.cpu().double()[:, None]


@pytest.mark.kernels_ci
@pytest.mark.parametrize("mx", [False, True])
@pytest.mark.parametrize("shape", [(96, 256), (1024, 5120), (130, 2048)])
def test_the_fp8_gemv_multiplies_by_the_stored_values_and_keeps_rows(mx, shape):
    n, k = shape
    c, s = fp8_matrix(n, k, mx)
    x = activations(16, k)
    values = fp8_values(c, s).to(dev)
    exact = x.double() @ values.t()
    magnitude = x.double().abs() @ values.abs().t()
    out = torch.empty(16, n, device=dev)
    nr.gemv_fp8(x, c, s, out)
    assert bool(((out.double() - exact).abs() <= magnitude * 2e-5 + 1e-30).all())
    for rows in range(1, 17):
        part = torch.empty(rows, n, device=dev)
        nr.gemv_fp8(x[:rows], c, s, part)
        assert torch.equal(part, out[:rows]), rows
        alone = torch.empty(1, n, device=dev)
        nr.gemv_fp8(x[rows - 1:rows].clone(), c, s, alone)
        assert torch.equal(alone[0], out[rows - 1]), rows
    small = (x[:3] / 64).half()   # FP16 out: keep the sums inside FP16's range
    half = torch.empty(3, n, dtype=torch.float16, device=dev)
    nr.gemv_fp8(small, c.view(torch.float8_e4m3fn), s, half)
    want = small.double() @ values.t()
    assert float((half.double() - want).abs().max() / want.abs().max()) < 2e-3


@pytest.mark.kernels_ci
def test_the_fp8_gemv_refuses_what_it_cannot_read():
    c, s = fp8_matrix(96, 256, False)
    out = torch.empty(2, 96, device=dev)
    with pytest.raises(RuntimeError):
        nr.gemv_fp8(activations(2, 256), c, s[:10], out)          # a scale per row
    cm, sm = fp8_matrix(96, 256, True)
    with pytest.raises(RuntimeError):
        nr.gemv_fp8(activations(2, 256), cm, sm[:, :4], out)      # an exponent per 32 inputs
    with pytest.raises(RuntimeError):
        nr.gemv_fp8(activations(2, 256).float(), c, s, out)      # 16-bit activations


@pytest.mark.kernels_ci
def test_every_triton_tile_gives_the_same_bits():
    """The autotuner picks a tile per shape by timing it: every tile it may pick gives the same bits, so two processes
    agree whatever they picked."""
    module = sys.modules[nr.gemv_triton.__module__]
    c, s, i = matrix(130, 2048)
    x = activations(8, 2048)
    found = []
    for config in module.GEMV_TILES:
        out = torch.empty(8, 130, device=dev)
        hw = module.hardware(dev)
        module._gemv_kernel.fn[(triton_cdiv(130, config.kwargs["BLOCK_N"]),)](
            x, c, s, i, out, 130, 2048, x.stride(0), out.stride(0), M=8, HW=hw, num_warps=config.num_warps,
            **config.kwargs)
        found.append(out)
    for out in found[1:]:
        assert torch.equal(out, found[0])


def triton_cdiv(a, b):
    return (a + b - 1) // b


def to_blocked(scales):
    """[rows][blocks] scale bytes in cuBLASLt's 128 x 4 tiled layout (rows padded to 128, blocks to 4)."""
    rows, cols = scales.shape
    rt, ct = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros(rt * 128, ct * 4, dtype=scales.dtype, device=scales.device)
    padded[:rows, :cols] = scales
    tiles = padded.view(rt, 128, ct, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


@pytest.mark.kernels_ci
@pytest.mark.parametrize("shape", [(96, 256), (130, 2080), (1000, 5120)])
def test_blocked_scales_give_the_row_major_bits(shape):
    n, k = shape
    c, s, i = matrix(n, k)
    x = activations(16, k)
    blocked = to_blocked(s.view(torch.uint8))
    for rows in (1, 5, 16):
        want = nvfp4_gemv(x[:rows], c, s, i)
        found = torch.empty_like(want)
        nr.gemv(x[:rows], c, blocked, i, found, blocked=True)
        assert torch.equal(found, want), (shape, rows)
    with pytest.raises(RuntimeError):
        nr.gemv(x[:2], c, blocked[:-4], i, torch.empty(2, n, device=dev), blocked=True)


@pytest.mark.kernels_ci
def test_tensors_on_another_device_or_misaligned_are_refused():
    c, s, i = matrix(96, 256)
    x = activations(2, 256)
    out = torch.empty(2, 96, device=dev)
    with pytest.raises(RuntimeError):
        nr.gemv(x, c, s, i.cpu(), out)
    odd = torch.empty(96 * 16 + 1, dtype=torch.uint8, device=dev)[1:].view(96, 16)
    odd.copy_(s.view(torch.uint8))
    with pytest.raises(RuntimeError):
        nr.gemv(x, c, odd, i, out)                       # scales read two bytes at a time
    fc, fs = fp8_matrix(96, 256, False)
    with pytest.raises(RuntimeError):
        nr.gemv_fp8(x, fc, fs.cpu(), out)
