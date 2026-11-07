"""fused-norms against a float64 reference: every function within FP32 rounding (FP32 out) or one output rounding
(16-bit out), weights per stream or head and the stream mixes too; each row's bits the same alone and beside others,
at any row count; strided views as their copies; out= and in-place rope; a CUDA graph replaying to the eager bits; the
two-pass path for rows wider than one block."""
import math

import pytest
import torch

import kernels

fn = kernels.get_kernel("crit-labs/fused-norms", version=1)
dev = torch.device("cuda")
FP32 = dict(atol=2e-5, rtol=2e-5)
STEP = {torch.bfloat16: 2.0 ** -8, torch.float16: 2.0 ** -11, torch.float32: 0.0}


def gen(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def normal(g, *shape, scale=1.0, dtype=torch.float32):
    return (torch.randn(*shape, generator=g) * scale).to(dev, dtype)


def inv_freq(rotary, theta):
    return (1.0 / theta ** (torch.arange(0, rotary, 2, dtype=torch.float64) / rotary)).float().to(dev)


def close(found, expected, dtype=torch.float32):
    found, expected = found.double(), expected.double()
    tol = FP32["atol"] + (FP32["rtol"] + STEP[dtype]) * expected.abs()
    if STEP[dtype]:
        tol = tol + STEP[dtype] * 0.01 * float(expected.abs().max())
    bad = (found - expected).abs() > tol
    assert not bad.any(), f"{int(bad.sum())} values off; worst {float((found - expected).abs().max())}"


# -- float64 references --------------------------------------------------------------------------------------------
def ref_rms(x, w, eps, offset=0.0):
    x = x.double()
    y = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * (offset + w.double())


def ref_rope(x, positions, freqs, rotary, style, scale=1.0):
    x = x.double()
    shape = x.shape
    x = x if x.dim() == 3 else x[:, None]
    half = rotary // 2
    if freqs.dim() == 1:
        angle = (positions.float()[:, None] * freqs[None, :]).double()   # the FP32 angle, as the contract names it
        cos, sin = angle.cos() * scale, angle.sin() * scale
    else:
        rows = freqs.double()[positions.clamp(0, freqs.shape[0] - 1)]
        cos, sin = rows[:, :half] * scale, rows[:, half:] * scale
    cos, sin = cos[:, None], sin[:, None]
    y = x.clone()
    if style == "half":
        a, b = x[..., :half], x[..., half:rotary]
        y[..., :half], y[..., half:rotary] = a * cos - b * sin, b * cos + a * sin
    else:
        a, b = x[..., 0:rotary:2], x[..., 1:rotary:2]
        y[..., 0:rotary:2], y[..., 1:rotary:2] = a * cos - b * sin, b * cos + a * sin
    return y.reshape(shape)


def ref_act(name, g):
    g = g.double()
    if name == "silu":
        return g * torch.sigmoid(g)
    if name == "gelu":
        return 0.5 * g * (1 + torch.erf(g / math.sqrt(2)))
    if name == "gelu_tanh":
        return 0.5 * g * (1 + torch.tanh(math.sqrt(2 / math.pi) * (g + 0.044715 * g ** 3)))
    return torch.sigmoid(g)


DTYPES = [torch.bfloat16, torch.float16, torch.float32]


# -- against the reference -----------------------------------------------------------------------------------------
@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", [64, 520, 2560, 5120, 7168, 20000])
def test_rms_norm(dtype, width):
    g = gen(width)
    x, r, w = normal(g, 5, width, scale=2, dtype=dtype), normal(g, 5, width, scale=2), normal(g, width) * 0.3 + 1
    for out_dtype in (torch.float32, dtype):
        close(fn.rms_norm(x, w, 1e-6, out_dtype=out_dtype), ref_rms(x, w, 1e-6), out_dtype)
        close(fn.rms_norm(x, w.to(dtype), 1e-5, offset=1.0, out_dtype=out_dtype), ref_rms(x, w.to(dtype), 1e-5, 1.0),
              out_dtype)
    close(fn.rms_norm(x, None, 1e-6, out_dtype=torch.float32), ref_rms(x, None, 1e-6))
    residual = r.clone()
    y, kept = fn.rms_norm(x, w, 1e-6, residual=residual, out_dtype=torch.float32)
    assert kept is residual
    close(kept, r.double() + x.double())
    close(y, ref_rms(r.double() + x.double(), w, 1e-6))
    low = r.to(dtype, copy=True)
    sums = low.double() + x.double()
    y, kept = fn.rms_norm(x, w, 1e-6, residual=low)
    assert y.dtype == dtype
    close(kept, sums, dtype)
    close(y, ref_rms(kept, w, 1e-6), dtype)   # the sums as the residual keeps them are what is normalized


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads,dim,rotary", [(24, 256, 64), (56, 128, 128), (2, 256, 64), (3, 96, 32), (4, 64, 64)])
def test_heads_rope_gates_l2(dtype, heads, dim, rotary):
    g = gen(heads * dim)
    x, z = normal(g, 7, heads, dim, scale=2, dtype=dtype), normal(g, 7, heads, dim, scale=2, dtype=dtype)
    w = normal(g, dim) * 0.3 + 1
    positions = torch.randint(0, 1 << 18, (7,), generator=g).to(dev)
    close(fn.rms_norm_heads(x, w, 1e-6, offset=0.5, out_dtype=torch.float32), ref_rms(x, w, 1e-6, 0.5))
    for style in ("half", "interleaved"):
        freqs = inv_freq(rotary, 1e7)
        expected = ref_rope(x, positions, freqs, rotary, style, 1.5)
        for out_dtype in (torch.float32, dtype):
            close(fn.rope(x, positions, freqs, rotary, style, 1.5, out_dtype=out_dtype), expected, out_dtype)
        close(fn.rms_norm_rope(x, w, 1e-6, positions, freqs, 0.5, rotary, style, out_dtype=torch.float32),
              ref_rope(ref_rms(x, w, 1e-6, 0.5), positions, freqs, rotary, style))
        p = torch.arange(64, dtype=torch.float64)[:, None] * freqs.double().cpu()[None, :]
        table = torch.cat([p.cos(), p.sin()], -1).float().to(dev)
        small = positions % 64
        close(fn.rope(x, small, table, style=style, out_dtype=torch.float32), ref_rope(x, small, table, rotary, style))
    for activation in ("silu", "sigmoid"):
        close(fn.gated_rms_norm(x, z, w, 1e-6, activation=activation, out_dtype=torch.float32),
              ref_rms(x, w, 1e-6) * ref_act(activation, z))
    xd = x.double()
    close(fn.l2_norm(x, 1e-6, out_dtype=torch.float32), xd / torch.sqrt(xd.pow(2).sum(-1, keepdim=True) + 1e-6))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", DTYPES)
def test_products(dtype):
    g = gen(1)
    x = normal(g, 6, 2 * 1500, scale=3, dtype=dtype)
    gate, up = x[:, :1500], x[:, 1500:]
    for out_dtype in (torch.float32, dtype):
        close(fn.silu_mul(x, out_dtype=out_dtype), ref_act("silu", gate) * up.double(), out_dtype)
    close(fn.silu_mul(x, layout="interleaved", out_dtype=torch.float32), ref_act("silu", x[:, 0::2]) * x[:, 1::2].double())
    close(fn.silu_mul(gate, up, out_dtype=torch.float32), ref_act("silu", gate) * up.double())
    close(fn.gelu_mul(x, out_dtype=torch.float32), ref_act("gelu", gate) * up.double())
    close(fn.gelu_mul(x, layout="interleaved", approximate="tanh", out_dtype=torch.float32),
          ref_act("gelu_tanh", x[:, 0::2]) * x[:, 1::2].double())
    close(fn.sigmoid_mul(up, gate, out_dtype=torch.float32), up.double() * torch.sigmoid(gate.double()))
    big = torch.tensor([[-100.0, 100.0, -30.0, 0.0]], device=dev)
    assert torch.isfinite(fn.silu_mul(big, big)).all() and torch.isfinite(fn.gelu_mul(big, big, approximate="tanh")).all()


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_weight_per_stream_and_the_stream_mixes(dtype):
    g = gen(7)
    streams = normal(g, 5, 4, 2560, scale=2, dtype=dtype)
    w = normal(g, 4, 2560) * 0.3
    close(fn.rms_norm(streams, w, 1e-6, offset=1.0, out_dtype=torch.float32), ref_rms(streams, w, 1e-6, 1.0))
    heads = normal(g, 3, 4, 64, scale=2, dtype=dtype)
    hw = normal(g, 4, 64) * 0.3 + 1
    positions = torch.randint(0, 4096, (3,), generator=g).to(dev)
    freqs = inv_freq(32, 1e4)
    close(fn.rms_norm_rope(heads, hw, 1e-6, positions, freqs, out_dtype=torch.float32),
          ref_rope(ref_rms(heads, hw, 1e-6), positions, freqs, 32, "half"))
    gates, per_stream = normal(g, 5, 4, 2560, scale=2, dtype=dtype), normal(g, 5, 4)
    sd = streams.double()
    for gate, activation, scale, factor in (
            (None, "none", 0.25, torch.ones_like(sd)), (gates, "sigmoid", 1.0, torch.sigmoid(gates.double())),
            (per_stream, "none", 1.0, per_stream.double()[:, :, None]),
            (per_stream, "silu", 0.5, ref_act("silu", per_stream)[:, :, None]), (gates, "none", 1.0, gates.double())):
        expected = scale * (factor * sd).sum(1)
        close(fn.stream_sum(streams, gate, activation, scale, out_dtype=torch.float32), expected)
    row = normal(g, 5, 2560, scale=2, dtype=dtype)
    for weights in (None, per_stream):
        kept = streams.float().clone()
        factor = 1.0 if weights is None else weights.double()[:, :, None]
        expected = kept.double() + factor * row.double()[:, None, :]
        assert fn.stream_add(kept, row, weights) is kept
        close(kept, expected)


# -- rows alone and beside others ----------------------------------------------------------------------------------
def calls(width, heads, dim, rotary):
    g = gen(width)
    n = 320
    x, r = normal(g, n, width, scale=2, dtype=torch.bfloat16), normal(g, n, width, scale=2)
    w = normal(g, width) * 0.3 + 1
    h, z = normal(g, n, heads, dim, scale=2, dtype=torch.bfloat16), normal(g, n, heads, dim, scale=2, dtype=torch.bfloat16)
    hw = normal(g, dim) * 0.3 + 1
    positions = torch.randint(0, 1 << 18, (n,), generator=g).to(dev)
    freqs = inv_freq(rotary, 1e7)
    gate_up = normal(g, n, 2 * width, scale=3, dtype=torch.bfloat16)
    streams, sw = normal(g, n, 4, width, scale=2), normal(g, 4, width) * 0.3 + 1
    gates, per_stream = normal(g, n, 4, width, scale=2, dtype=torch.bfloat16), normal(g, n, 4)

    def added(k):
        kept = streams[:k].clone()
        return fn.stream_add(kept, x[:k].clone(), per_stream[:k].clone())

    return {
        "rms_norm": lambda k: fn.rms_norm(x[:k].clone(), w, 1e-6),
        "rms_norm residual": lambda k: torch.cat([t.float() for t in fn.rms_norm(
            x[:k].clone(), w, 1e-6, offset=1.0, residual=r[:k].clone(), out_dtype=torch.bfloat16)], -1),
        "rms_norm_heads": lambda k: fn.rms_norm_heads(h[:k].clone(), hw, 1e-6, offset=1.0),
        "rms_norm_rope": lambda k: fn.rms_norm_rope(h[:k].clone(), hw, 1e-6, positions[:k].clone(), freqs),
        "rope": lambda k: fn.rope(h[:k].clone(), positions[:k].clone(), freqs, style="interleaved"),
        "gated_rms_norm": lambda k: fn.gated_rms_norm(h[:k].clone(), z[:k].clone(), hw, 1e-6),
        "l2_norm": lambda k: fn.l2_norm(h[:k].clone()),
        "silu_mul": lambda k: fn.silu_mul(gate_up[:k].clone()),
        "gelu_mul": lambda k: fn.gelu_mul(gate_up[:k].clone(), layout="interleaved"),
        "sigmoid_mul": lambda k: fn.sigmoid_mul(x[:k].clone(), gate_up[:k, :width].clone()),
        "rms_norm per stream": lambda k: fn.rms_norm(streams[:k].clone(), sw, 1e-6, offset=1.0, out_dtype=torch.bfloat16),
        "stream_sum": lambda k: fn.stream_sum(streams[:k].clone(), gates[:k].clone(), "sigmoid", 0.25),
        "stream_add": added,
    }


@pytest.mark.kernels_ci
@pytest.mark.parametrize("width,heads,dim,rotary", [(5120, 24, 256, 64), (7168, 56, 128, 128), (2560, 2, 256, 64),
                                                    (20000, 4, 64, 16)])
def test_a_row_has_its_bits_alone_and_beside_others(width, heads, dim, rotary):
    for name, call in calls(width, heads, dim, rotary).items():
        whole = call(320)
        for k in (1, 2, 3, 4, 5, 7, 8, 64):
            assert torch.equal(call(k), whole[:k]), f"{name}: a {k}-row call differs from the same rows of 320"


@pytest.mark.kernels_ci
def test_views_give_the_bits_of_their_copies():
    g = gen(3)
    qg = normal(g, 6, 4, 128, dtype=torch.bfloat16)
    q, gate = qg[..., :64], qg[..., 64:]
    w = normal(g, 64) * 0.2 + 1
    positions = torch.randint(0, 4096, (6,), generator=g).to(dev)
    freqs = inv_freq(16, 1e4)
    for call in (lambda t: fn.rms_norm_heads(t, w, 1e-6, offset=1.0),
                 lambda t: fn.rms_norm_rope(t, w, 1e-6, positions, freqs, offset=1.0),
                 lambda t: fn.rope(t, positions, freqs), lambda t: fn.l2_norm(t),
                 lambda t: fn.sigmoid_mul(t, gate), lambda t: fn.gated_rms_norm(t, gate, w, 1e-6)):
        assert torch.equal(call(q), call(q.contiguous()))
    fused = normal(g, 6, 1536, dtype=torch.bfloat16)
    wide = normal(g, 512) * 0.2 + 1
    assert torch.equal(fn.rms_norm(fused[:, 512:1024], wide, 1e-6), fn.rms_norm(fused[:, 512:1024].contiguous(), wide, 1e-6))
    assert torch.equal(fn.silu_mul(fused[:, :512], fused[:, 1024:]),
                       fn.silu_mul(fused[:, :512].contiguous(), fused[:, 1024:].contiguous()))
    split = normal(g, 6, 2, 192, dtype=torch.bfloat16)
    x4, z4 = split[..., :64].view(6, 2, 2, 32), split[..., 64:128].view(6, 2, 2, 32)
    hw = normal(g, 32) * 0.2 + 1
    assert torch.equal(fn.gated_rms_norm(x4, z4, hw, 1e-6), fn.gated_rms_norm(x4.contiguous(), z4.contiguous(), hw, 1e-6))
    odd = normal(g, 4, 6, 3, 32, dtype=torch.bfloat16).transpose(1, 2)   # leading dimensions no two levels cover
    assert torch.equal(fn.l2_norm(odd), fn.l2_norm(odd.contiguous()))


@pytest.mark.kernels_ci
def test_out_in_place_and_dtypes():
    g = gen(5)
    x = normal(g, 3, 2, 64, dtype=torch.bfloat16)
    positions = torch.randint(0, 4096, (3,), generator=g).to(dev)
    freqs = inv_freq(64, 1e4)
    expected = fn.rope(x, positions, freqs)
    out = torch.empty_like(x)
    assert fn.rope(x, positions, freqs, out=out) is out and torch.equal(out, expected)
    inplace = x.clone()
    assert fn.rope(inplace, positions, freqs, out=inplace) is inplace and torch.equal(inplace, expected)
    w = normal(g, 64) + 1
    for out_dtype in DTYPES:
        assert fn.rms_norm(x, w, 1e-6, out_dtype=out_dtype).dtype == out_dtype
    assert tuple(fn.silu_mul(x.view(3, 128)).shape) == (3, 64)
    assert tuple(fn.silu_mul(x, layout="interleaved").shape) == (3, 2, 32)
    for bad in (lambda: fn.rms_norm(x, normal(g, 32), 1e-6), lambda: fn.silu_mul(x.view(3, 128)[:, :127]),
                lambda: fn.rope(x, positions[:2], freqs), lambda: fn.rope(x, positions, inv_freq(80, 1e4)),
                lambda: fn.gated_rms_norm(x, x[:2], w, 1e-6), lambda: fn.rms_norm(x.cpu(), None, 1e-6)):
        with pytest.raises(ValueError):
            bad()


@pytest.mark.kernels_ci
def test_a_captured_graph_replays_the_eager_bits():
    g = gen(9)
    x, residual = normal(g, 3, 2560, dtype=torch.bfloat16), normal(g, 3, 2560)
    w = normal(g, 2560) + 1
    h, z = normal(g, 3, 4, 128, dtype=torch.bfloat16), normal(g, 3, 4, 128, dtype=torch.bfloat16)
    hw = normal(g, 128) + 1
    positions = torch.randint(0, 1 << 16, (3,), generator=g).to(dev)
    freqs = inv_freq(32, 1e6)
    gate_up = normal(g, 3, 5120, dtype=torch.bfloat16)

    def run():
        y, kept = fn.rms_norm(x, w, 1e-6, offset=1.0, residual=residual, out_dtype=torch.bfloat16)
        return [y, kept.clone(), fn.rms_norm_heads(h, hw, 1e-6), fn.rms_norm_rope(h, hw, 1e-6, positions, freqs),
                fn.rope(h, positions, freqs, style="interleaved"), fn.gated_rms_norm(h, z, hw, 1e-6), fn.l2_norm(h),
                fn.silu_mul(gate_up), fn.gelu_mul(gate_up, approximate="tanh"), fn.sigmoid_mul(x, gate_up[:, :2560])]

    start = residual.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    residual.copy_(start)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    g2 = gen(10)
    for t in (x, residual, w, h, z, hw, gate_up):
        t.copy_(normal(g2, *t.shape, dtype=t.dtype))
    positions.copy_(torch.randint(0, 1 << 16, (3,), generator=g2).to(dev))
    start = residual.clone()
    graph.replay()
    replayed = [t.clone() for t in captured]
    residual.copy_(start)
    for a, b in zip(replayed, run()):
        assert torch.equal(a, b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
