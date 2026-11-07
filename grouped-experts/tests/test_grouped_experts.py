"""The grouped experts against an FP64 reference of the same decoded weights in every format, and the bits a route
keeps: the same alone and beside other tokens, in decode-width (8) and prompt-width (32) items, through the one-block
plan and the sorted one, on a repeat, and in a captured CUDA graph. The bundles are packed here in crit's
experts:bundles layout (crit's inference/grouped-experts code/bundles.py is the definition)."""
import math

import pytest
import torch

import kernels

gx = kernels.get_kernel("crit-labs/grouped-experts", version=1)
dev = torch.device("cuda")
FORMATS = {"bf16": 0, "fp8:channel": 1, "fp8:block": 2, "nvfp4": 3, "int4": 4, "mxfp4": 5}
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_SIGNED = torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float64)


def cdiv(a, b):
    return -(-a // b)


def components(fmt, H, I):
    if fmt == "bf16":
        return [("gate_up", 4 * I * H), ("down", 2 * H * I)]
    if fmt.startswith("fp8"):
        gs, ds = ((8 * I, 4 * H) if fmt == "fp8:channel" else
                  (4 * cdiv(2 * I, 128) * cdiv(H, 128), 4 * cdiv(H, 128) * cdiv(I, 128)))
        return [("gate_up", 2 * I * H), ("gate_up_scale", gs), ("down", H * I), ("down_scale", ds)]
    if fmt in ("nvfp4", "int4"):
        return [("gate_up", I * H), ("gate_up_scale", I * H // 8), ("gate_up_inverse", 8 * I), ("down", H * I // 2),
                ("down_scale", H * I // 16), ("down_inverse", 4 * H), ("input", 16)]
    return [("gate_up", I * H), ("gate_up_scale", I * H // 16), ("down", H * I // 2), ("down_scale", H * I // 32)]


def offsets(fmt, H, I):
    found, at = {}, 0
    for name, size in components(fmt, H, I):
        found[name] = at
        at += cdiv(size, 16) * 16
    return found, at


def layout(fmt, H, I):
    o, size = offsets(fmt, H, I)
    keys = ("gate_up", "gate_up_scale", "gate_up_inverse", "down", "down_scale", "down_inverse", "input")
    return [FORMATS[fmt], size] + [o.get(k, -1) for k in keys] + [H, I]


def e4m3_bytes_of(values):
    return values.float().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)


def matrix(fmt, rows, cols, g):
    """(component tensors, decoded FP64 values) of one random matrix."""
    if fmt == "bf16":
        w = (torch.randn(rows, cols, generator=g) * 0.05).bfloat16()
        return {"": w.view(torch.uint8).reshape(-1)}, w.double()
    if fmt.startswith("fp8"):
        codes = e4m3_bytes_of(torch.randn(rows, cols, generator=g) * 64)
        v = codes.view(torch.float8_e4m3fn).double()
        if fmt == "fp8:channel":
            s = torch.rand(rows, generator=g) * 0.01 + 0.0005
            return {"": codes.reshape(-1), "scale": s.float()}, v * s.double()[:, None]
        s = torch.rand(cdiv(rows, 128), cdiv(cols, 128), generator=g) * 0.01 + 0.0005
        full = s.double().repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows, :cols]
        return {"": codes.reshape(-1), "scale": s.float()}, v * full
    codes = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g)
    lo, hi = (codes & 15).long(), (codes >> 4).long()
    if fmt == "int4":
        int4 = torch.tensor([float(c if c < 8 else c - 16) for c in range(16)], dtype=torch.float64)
        vals = torch.stack([int4[lo], int4[hi]], -1).reshape(rows, cols)
    else:
        vals = torch.stack([E2M1_SIGNED[lo], E2M1_SIGNED[hi]], -1).reshape(rows, cols)
    if fmt == "mxfp4":
        s = torch.randint(120, 127, (rows, cols // 32), dtype=torch.uint8, generator=g)
        return {"": codes.reshape(-1), "scale": s}, vals * torch.pow(2.0, s.double() - 127).repeat_interleave(32, 1)
    exp, man = torch.randint(5, 9, (rows, cols // 16), generator=g), torch.randint(0, 8, (rows, cols // 16), generator=g)
    s = (exp * 8 + man).to(torch.uint8)
    inverse = torch.rand(rows, generator=g) * 0.05 + 0.01
    decoded = vals * s.view(torch.float8_e4m3fn).double().repeat_interleave(16, 1) * inverse.double()[:, None]
    return {"": codes.reshape(-1), "scale": s, "inverse": inverse.float()}, decoded


def bundle(fmt, H, I, g, inputs=(0.0, 0.0)):
    """One expert's bundle (uint8 on the device) and its decoded FP64 gate/up and down matrices."""
    o, size = offsets(fmt, H, I)
    out = torch.zeros(size, dtype=torch.uint8)
    gu, gu_values = matrix(fmt, 2 * I, H, g)
    dn, dn_values = matrix(fmt, H, I, g)
    for prefix, parts in (("gate_up", gu), ("down", dn)):
        for suffix, t in parts.items():
            name = prefix + (f"_{suffix}" if suffix else "")
            raw = t.contiguous().view(torch.uint8).reshape(-1)
            out[o[name]:o[name] + raw.numel()] = raw
    if "input" in o:
        a, b = inputs
        values = torch.tensor([a, b, 1 / a if a else 0.0, 1 / b if b else 0.0], dtype=torch.float32)
        out[o["input"]:o["input"] + 16] = values.view(torch.uint8)
    return out, gu_values, dn_values


def quantize(v, g):
    """crit's declared NVFP4 quantizer as values (e2m1 * decoded / g), FP32 steps, per row of 16-blocks."""
    signed, decoded = quantize_codes(v, g)
    return (signed.double() * decoded.double()[..., None] / g).reshape(v.shape)


def quantize_codes(v, g):
    """(signed E2M1 magnitudes [rows][blocks][16] FP32, decoded block scales [rows][blocks] FP32)."""
    rows, cols = v.shape
    blocks = v.float().reshape(rows, cols // 16, 16)
    amax = blocks.abs().amax(-1)
    wanted = (amax * g) / 6.0
    wanted = torch.where(torch.isfinite(wanted), wanted, torch.full_like(wanted, 448.0))
    decoded = wanted.clamp(max=448.0).to(torch.float8_e4m3fn).float()
    q = (blocks * g) / decoded[..., None].clamp(min=1e-38)
    a = q.abs()
    code = sum((a > b).int() if strict else (a >= b).int()
               for b, strict in ((0.25, 1), (0.75, 0), (1.25, 1), (1.75, 0), (2.5, 1), (3.5, 0), (5.0, 1)))
    mag = torch.tensor(E2M1, dtype=torch.float32)[code.long()]
    signed = torch.where((q < 0) & (code > 0), -mag, mag)
    return torch.where(decoded[..., None] > 0, signed, torch.zeros_like(signed)), decoded


class Experts:
    def __init__(self, fmt, E=6, H=256, I=128, seed=0, a4=False):
        g = torch.Generator().manual_seed(seed)
        self.fmt, self.E, self.H, self.I, self.a4 = fmt, E, H, I, a4
        self.inputs = [(float(torch.rand(1, generator=g)) * 200 + 300, float(torch.rand(1, generator=g)) * 40 + 20)
                       for _ in range(E)]
        made = [bundle(fmt, H, I, g, self.inputs[e]) for e in range(E)]
        self.store = torch.stack([m[0] for m in made]).to(dev)
        self.gate_up, self.down = [m[1] for m in made], [m[2] for m in made]
        self.layout = layout(fmt, H, I)

    def reference(self, x, ids, weights, slots=None):
        x = x.bfloat16().double().cpu()
        out = torch.zeros(x.shape[0], self.H, dtype=torch.float64)
        for t in range(x.shape[0]):
            for j in range(ids.shape[1]):
                e = int(ids[t, j])
                if e < 0 or (slots is not None and int(slots[e]) < 0):
                    continue
                xin = quantize(x[t:t + 1].float(), self.inputs[e][0])[0] if self.a4 else x[t]
                gu = self.gate_up[e] @ xin
                h = torch.nn.functional.silu(gu[:self.I]) * gu[self.I:]
                h = quantize(h[None].float(), self.inputs[e][1])[0] if self.a4 else h.float().bfloat16().double()
                out[t] += float(weights[t, j]) * (self.down[e] @ h)
        return out

    def forward(self, x, ids, weights, slots=None, width=None):
        ws = gx.Workspace(ids.shape[0], ids.shape[1], self.E, self.H, self.I, dev, width)
        return gx.forward(x, ids, weights, self.store, slots, self.layout, ws, self.a4)

    def rows(self, x, ids, slots=None, width=None):
        ws = gx.Workspace(ids.shape[0], ids.shape[1], self.E, self.H, self.I, dev, width)
        return gx.route_outputs(x, ids, self.store, slots, self.layout, ws, self.a4).clone()


def routes(T, top, E, g, holes=True):
    ids = torch.stack([torch.randperm(E, generator=g)[:top] for _ in range(T)])
    if holes:
        ids[torch.rand(T, top, generator=g) < 0.1] = -1
    w = torch.rand(T, top, generator=g)
    return ids.to(dev), w.to(dev)


CASES = [("bf16", False), ("fp8:channel", False), ("fp8:block", False), ("nvfp4", False), ("nvfp4", True),
         ("int4", False), ("mxfp4", False)]


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fmt,a4", CASES)
def test_every_format_matches_the_fp64_reference(fmt, a4):
    experts = Experts(fmt, a4=a4)
    g = torch.Generator().manual_seed(3)
    x = (torch.randn(13, experts.H, generator=g)).bfloat16().to(dev)
    ids, w = routes(13, 3, experts.E, g)
    out = experts.forward(x, ids, w).double().cpu()
    ref = experts.reference(x, ids.cpu(), w.cpu())
    scale = ref.abs().max()
    assert (out - ref).abs().max() <= (3e-2 if a4 else 5e-3) * scale, (fmt, a4, float((out - ref).abs().max() / scale))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fmt,a4", [("bf16", False), ("nvfp4", True), ("fp8:block", False), ("mxfp4", False)])
def test_a_route_keeps_its_bits_whatever_shares_its_call(fmt, a4):
    experts = Experts(fmt, E=16, a4=a4, seed=1)
    g = torch.Generator().manual_seed(5)
    T, top = 300, 4   # 1200 routes: the sorted plan; the slices below take the one-block plan
    x = torch.randn(T, experts.H, generator=g).bfloat16().to(dev)
    ids, w = routes(T, top, experts.E, g)
    full = experts.rows(x, ids)
    valid = (ids.reshape(-1) >= 0).cpu()
    for width in (8, 32):
        for first, count in ((0, 1), (17, 1), (40, 7), (100, 200)):
            part = experts.rows(x[first:first + count].clone(), ids[first:first + count].clone(), width=width)
            sel = valid[first * top:(first + count) * top]
            assert torch.equal(part[sel], full[first * top:(first + count) * top][sel]), (fmt, width, first, count)
    again = experts.rows(x, ids)
    assert torch.equal(again[valid], full[valid])
    out = experts.forward(x, ids, w)
    assert torch.equal(experts.forward(x[17:18].clone(), ids[17:18].clone(), w[17:18].clone())[0], out[17])


@pytest.mark.kernels_ci
def test_slots_place_experts_anywhere_and_negative_slots_are_skipped():
    experts = Experts("nvfp4", E=8, seed=2)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(20, experts.H, generator=g).bfloat16().to(dev)
    ids, w = routes(20, 3, experts.E, g, holes=False)
    base = experts.forward(x, ids, w)
    perm = torch.randperm(experts.E, generator=g)
    store = torch.zeros(experts.E + 3, experts.store.shape[1], dtype=torch.uint8, device=dev)
    store[perm + 3] = experts.store
    slots = (perm + 3).int().to(dev)
    ws = gx.Workspace(20, 3, experts.E, experts.H, experts.I, dev)
    moved = gx.forward(x, ids, w, store, slots, experts.layout, ws)
    assert torch.equal(moved, base)
    slots[2] = -2   # expert 2 on another tier: its routes skipped here
    skipped = gx.forward(x, ids, w, store, slots, experts.layout, ws)
    ref = experts.reference(x, ids.cpu(), w.cpu(), slots.cpu())
    assert (skipped.double().cpu() - ref).abs().max() <= 5e-3 * ref.abs().max()
    y = gx.route_outputs(x, ids, store, slots, experts.layout, ws).clone()
    full = experts.rows(x, ids)
    tier = (ids.reshape(-1) == 2)
    y[tier] = full[tier]   # the other tier's rows, then the declared combine without slots
    assert torch.equal(gx.combine(y, ids, w, None, experts.E), base)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("tokens", [1, 8, 120])
def test_a_captured_call_replays_new_routes(tokens):
    experts = Experts("nvfp4", E=32, seed=4, a4=True)
    g = torch.Generator().manual_seed(9)
    x = torch.randn(tokens, experts.H, generator=g).bfloat16().to(dev)
    ids, w = routes(tokens, 4, experts.E, g)
    ws = gx.Workspace(tokens, 4, experts.E, experts.H, experts.I, dev)
    out = torch.empty(tokens, experts.H, device=dev)
    gx.forward(x, ids, w, experts.store, None, experts.layout, ws, True, out)   # the first call, outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gx.forward(x, ids, w, experts.store, None, experts.layout, ws, True, out)
    for seed in (11, 12):
        g2 = torch.Generator().manual_seed(seed)
        x.copy_(torch.randn(tokens, experts.H, generator=g2).bfloat16())
        new_ids, new_w = routes(tokens, 4, experts.E, g2)
        ids.copy_(new_ids)
        w.copy_(new_w)
        graph.replay()
        eager = experts.forward(x, ids, w)
        assert torch.equal(out, eager)


@pytest.mark.kernels_ci
def test_the_quantizer_is_the_declared_one():
    g = torch.Generator().manual_seed(13)
    x = torch.randn(9, 256, generator=g) * torch.rand(9, 1, generator=g) * 10
    x[0, :16] = 0.0
    x[1, 3] = float("nan")
    values, scales = gx.quantize_e2m1x2(x.to(dev), 37.5)
    signed, decoded = quantize_codes(torch.nan_to_num(x, nan=0.0), 37.5)
    assert torch.equal(values.cpu(), (2 * signed).to(torch.int8).reshape(x.shape))
    assert torch.equal(scales.cpu(), decoded / torch.tensor(37.5))
