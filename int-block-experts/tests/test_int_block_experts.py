"""The decode rows and the prompt GEMM give the same bits for every output, on a repeat too.
tests/int_block_tests.cu checks the raw kernels against the declared order's scalar oracle in C++."""
import pytest
import torch

import kernels

ib = kernels.get_kernel("crit-labs/int-block-experts", version=1)
dev = torch.device("cuda")


@pytest.mark.kernels_ci
def test_rows_and_gemm_agree_bit_for_bit():
    g = torch.Generator().manual_seed(0)
    rows, cols, tokens = 96, 704, 21
    codes = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g).to(dev)
    scales = (torch.randint(0x20, 0x5F, (rows, cols // 16), generator=g)).to(torch.uint8).to(dev)
    x = torch.randn(tokens, cols, generator=g).to(dev)
    values, act_scales = ib.quantize_int8(x, 32)
    table = ib.matrix_table([(codes, scales, 0.5, False)])
    decode = torch.zeros(tokens, rows, device=dev)
    ib.rows(table, ib.row_jobs([(0, list(range(t, min(t + 8, tokens))), list(range(t, min(t + 8, tokens))))
                                for t in range(0, tokens, 8)], dev), rows, values, act_scales, 32, decode)
    prompt = torch.zeros(tokens, rows, device=dev)
    ib.gemm(table, ib.gemm_jobs([(0, 0, tokens)], dev), rows, tokens, values, act_scales, 32, prompt)
    assert torch.equal(decode.view(torch.int32), prompt.view(torch.int32))
    again = torch.zeros_like(prompt)
    ib.gemm(table, ib.gemm_jobs([(0, 0, tokens)], dev), rows, tokens, values, act_scales, 32, again)
    assert torch.equal(again, prompt)
