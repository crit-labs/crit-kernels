"""The decode rows and the prompt GEMM give the same bits for every output, on a repeat too.
tests/fp8_block_tests.cu checks the raw kernels against the declared order's reference kernel in C++."""
import pytest
import torch

import kernels

fb = kernels.get_kernel("crit-labs/fp8-block-rows", version=1)
dev = torch.device("cuda")


@pytest.mark.kernels_ci
def test_rows_and_gemm_agree_bit_for_bit():
    g = torch.Generator().manual_seed(0)
    rows, cols, tokens = 96, 704, 21
    codes = torch.randint(0, 0x7E, (rows, cols), dtype=torch.uint8, generator=g)
    codes = (codes | (torch.randint(0, 2, (rows, cols), generator=g) * 0x80).to(torch.uint8)).to(dev)
    scales = torch.randint(120, 134, (rows, cols // 32), dtype=torch.uint8, generator=g).to(dev)  # E8M0 bytes
    x = torch.randn(tokens, cols, generator=g).to(dev, torch.bfloat16)
    act_codes, act_scales = fb.quantize(x, 32, fb.MX)
    table = fb.matrix_table([(codes, scales, 1, 32, 0.5)])  # the table holds pointers: keep the tensors alive
    decode = torch.zeros(tokens, rows, device=dev)
    fb.rows(table, fb.row_jobs([(0, list(range(t, min(t + 8, tokens))), list(range(t, min(t + 8, tokens))))
                                for t in range(0, tokens, 8)], dev), rows, act_codes, act_scales, 32, decode)
    prompt = torch.zeros(tokens, rows, device=dev)
    fb.gemm(table, fb.gemm_jobs([(0, 0, tokens)], dev), rows, tokens, act_codes, act_scales, 32, prompt)
    assert torch.equal(decode.view(torch.int32), prompt.view(torch.int32))
    again = torch.zeros_like(prompt)
    fb.gemm(table, fb.gemm_jobs([(0, 0, tokens)], dev), rows, tokens, act_codes, act_scales, 32, again)
    assert torch.equal(again, prompt)
