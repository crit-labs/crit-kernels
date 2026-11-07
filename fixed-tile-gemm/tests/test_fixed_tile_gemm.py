"""The fixed-tile GEMM: x times dense within FP32 accumulation of the FP64 product, and a row's result the same bits
whatever rows share the call, across the switch between the large and the small tile."""
import pytest
import torch

import kernels

ftg = kernels.get_kernel("crit-labs/fixed-tile-gemm", version=1)
dev = torch.device("cuda")


def operands(m, n, k, dtype=torch.bfloat16, seed=4):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(m, k, generator=g).to(dev, dtype), (torch.randn(n, k, generator=g) * 0.05).to(dev, dtype))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_it_multiplies_within_fp32_accumulation_of_the_fp64_product(dtype):
    x, w = operands(300, 96, 256, dtype)
    exact = x.double() @ w.double().t()
    found = ftg.mm(x, w, torch.float32).double()
    assert float((found - exact).abs().max() / exact.abs().max()) < 1e-5
    assert ftg.mm(x, w).dtype == dtype


@pytest.mark.kernels_ci
def test_a_row_keeps_its_bits_whatever_rows_share_the_call():
    # 45,056 rows of a 96-row matrix fill 352 large tiles (the large one on a device of up to 176 multiprocessors);
    # the slices below fill too few and take the small one: the switch keeps every row's order.
    rows = 45056
    x, w = operands(rows, 96, 256)
    whole = ftg.mm(x, w)
    for first, count in ((0, 1), (0, 9), (17, 40), (300, 700), (rows - 9, 9)):
        assert torch.equal(ftg.mm(x[first:first + count].clone(), w), whole[first:first + count]), (first, count)
    assert torch.equal(ftg.mm(x, w), whole)   # and on a repeat


@pytest.mark.kernels_ci
def test_real_widths_keep_rows_too():
    x, w = operands(1024, 5120, 2048)
    whole = ftg.mm(x, w)
    for first, count in ((0, 9), (511, 64), (1000, 24)):
        assert torch.equal(ftg.mm(x[first:first + count].clone(), w), whole[first:first + count])


def test_it_refuses_mixed_or_unsupported_dtypes():
    x, w = operands(16, 32, 64)
    with pytest.raises(ValueError):
        ftg.mm(x, w.half())
    with pytest.raises(ValueError):
        ftg.mm(x.float(), w.float())
