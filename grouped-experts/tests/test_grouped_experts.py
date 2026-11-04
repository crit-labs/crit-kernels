"""moe_block against an FP64 reference of the same decoded weights, same bits on a repeat, and a token's result
independent of its chunk-mates. tests/grouped_experts_tests.cu checks the raw kernels the same way in C++."""
import pytest
import torch

import kernels

gx = kernels.get_kernel("crit-labs/grouped-experts", version=1)
dev = torch.device("cuda")
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)


def nvfp4(rows, cols, g):
    """Random NVFP4 codes and E4M3 block scales (exponents 5..8) and their decoded FP64 values (post 0.05)."""
    codes = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g)
    exp = torch.randint(5, 9, (rows, cols // 16), generator=g)
    man = torch.randint(0, 8, (rows, cols // 16), generator=g)
    scales = (exp * 8 + man).to(torch.uint8)
    scale = (8 + man).double() * torch.pow(2.0, (exp - 10).double())
    lo, hi = E2M1[(codes & 15).long()], E2M1[(codes >> 4).long()]
    values = torch.stack([lo, hi], dim=-1).reshape(rows, cols) * scale.repeat_interleave(16, dim=1) * 0.05
    return codes.to(dev), scales.to(dev), values


@pytest.mark.kernels_ci
def test_moe_block_matches_the_fp64_reference_and_repeats():
    g = torch.Generator().manual_seed(0)
    T, K, E, H, I = 150, 4, 12, 128, 64
    gate_up, down, keep = [], [], []
    for e in range(E):
        gu, dn = nvfp4(2 * I, H, g), nvfp4(H, I, g)
        keep += [gu, dn]
        gate_up.append((gu[0], gu[1], 0.05))
        down.append((dn[0], dn[1], 0.05))
    t_gu, t_dn = gx.expert_table(gate_up), gx.expert_table(down)
    x = (torch.rand(T, H, generator=g) * 2 - 1).half().to(dev)
    ids = torch.randint(0, E - 1, (T, K), generator=g, dtype=torch.int32).to(dev)
    w = torch.rand(T, K, generator=g).to(dev)
    out = gx.moe_block(x, ids, w, "nvfp4", t_gu, t_dn, I)
    assert torch.equal(out, gx.moe_block(x, ids, w, "nvfp4", t_gu, t_dn, I))
    head = gx.moe_block(x[:70], ids[:70], w[:70], "nvfp4", t_gu, t_dn, I)
    assert torch.equal(out[:70], head)
    ref = torch.zeros(T, H, dtype=torch.float64)
    xd = x.double().cpu()
    for t in range(T):
        for j in range(K):
            e = int(ids[t, j])
            gu, dn = keep[2 * e][2], keep[2 * e + 1][2]
            h = gu @ xd[t]
            act = torch.nn.functional.silu(h[:I]) * h[I:]
            ref[t] += float(w[t, j]) * (dn @ act)
    assert (out.double().cpu() - ref).abs().max() <= 2e-2 * ref.abs().max()
