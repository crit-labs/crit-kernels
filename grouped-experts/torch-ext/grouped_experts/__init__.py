"""Routed SwiGLU experts over a slot store of expert bundles, for decode rows and prompt chunks alike (crit's
grouped-experts plugin, inference/grouped-experts, loads this from the kernel registry).

An expert is one bundle in crit's experts:bundles layout; a call names each bundle's components by byte offset (a
layout: [format, bytes, gate_up, gate_up_scale, gate_up_inverse, down, down_scale, down_inverse, input, hidden,
inter]), the store holding the bundles ([slots][bytes] uint8) and each expert's slot (slots [experts] int32: >= 0 a
slot, negative skipped). The routes are grouped by expert on the device (no host synchronization), every expert's
gate/up and down rows run on BF16 tensor cores with the weights decoded exactly, and each token's routes are summed in
route order. A route's result is the same bits whatever rows, chunk or item size share its call, so decode rows and
prompt chunks compute one function, and a call at a fixed token count can be captured in a CUDA graph once its
Workspace exists.

    ws = Workspace(tokens, top, experts, hidden, inter, x.device)
    out = forward(x, ids, weights, store, slots, layout, ws)          # [tokens][hidden] FP32
    y = route_outputs(x, ids, store, slots, layout, ws)               # unweighted rows, [tokens * top][hidden]
    out = combine(y, ids, weights, slots, experts)                    # fmaf(weight, row, sum) in route order
"""
import math

import torch

from ._ops import ops

__all__ = ["FORMATS", "SMALL_PLAN", "Workspace", "combine", "forward", "quantize_e2m1x2", "route_outputs",
           "width_for"]

FORMATS = {"bf16": 0, "fp8:channel": 1, "fp8:block": 2, "nvfp4": 3, "int4": 4, "mxfp4": 5}
SMALL_PLAN = 1024   # routes the one-block plan takes; larger calls group their routes from a stable sort


def width_for(routes, experts):
    """Routes per item: 32 once a call averages 8 or more routes per expert (prompt chunks), else 8 (decode rows)."""
    return 32 if routes >= 8 * experts else 8


class Workspace:
    """The buffers of calls of `tokens` tokens with `top` routes each over `experts` experts of a [hidden][inter]
    shape: the items, the intermediate h and the unweighted route rows y. Allocated once; reuse it for every call of
    the shape (and capture after one call)."""

    def __init__(self, tokens, top, experts, hidden, inter, device, width=None):
        routes = tokens * top
        self.tokens, self.top, self.experts, self.routes = tokens, top, experts, routes
        self.width = width or width_for(routes, experts)
        if self.width not in (8, 32):
            raise ValueError("items hold 8 or 32 routes")
        self.capacity = math.ceil(routes / self.width) + min(experts, routes)
        if self.capacity > 65535:
            raise ValueError(f"{routes} routes make more than 65535 items: split the call")
        ints = dict(dtype=torch.int32, device=device)
        self.item_slot = torch.zeros(self.capacity, **ints)
        self.item_routes = torch.full((self.capacity, self.width), -1, **ints)
        self.item_count = torch.zeros(1, **ints)
        self.h = torch.empty(routes, inter, dtype=torch.bfloat16, device=device)
        self.y = torch.zeros(routes, hidden, dtype=torch.float32, device=device)
        self.ones = torch.ones(routes, **ints) if routes > SMALL_PLAN else None


def _plan(ids, slots, ws):
    flat = ids.reshape(-1)
    if flat.numel() != ws.routes:
        raise ValueError(f"the workspace takes {ws.routes} routes, not {flat.numel()}")
    if ws.routes <= SMALL_PLAN:
        ops.plan(flat, slots, ws.experts, ws.item_slot, ws.item_routes, ws.item_count)
        return
    experts = ws.experts
    valid = (flat >= 0) & (flat < experts)
    if slots is not None:
        valid &= slots[flat.clamp(0, experts - 1)] >= 0
    keys = torch.where(valid, flat.long(), experts)
    order = torch.argsort(keys, stable=True)
    counts = torch.zeros(experts + 1, dtype=torch.int32, device=flat.device).scatter_add_(0, keys, ws.ones)
    starts = torch.cumsum(counts, 0, dtype=torch.int32) - counts
    item_end = torch.cumsum((counts[:experts] + ws.width - 1) // ws.width, 0, dtype=torch.int32)
    ops.items(order, counts, starts, item_end, slots, experts, ws.item_slot, ws.item_routes, ws.item_count)


def route_outputs(x, ids, store, slots, layout, ws, a4=False):
    """Each planned route's unweighted output row, y [tokens * top][hidden] FP32 (ws.y; rows of skipped routes keep
    what they held). x [tokens][hidden] BF16 contiguous; ids [tokens][top]; store uint8 [slots][bytes]; slots int32
    [experts] or None (expert e in slot e); a4: nvfp4 W4A4 at the bundles' input scales."""
    if ids.shape != (ws.tokens, ws.top):
        raise ValueError(f"the workspace takes ids [{ws.tokens}][{ws.top}], not {list(ids.shape)}")
    _plan(ids, slots, ws)
    ops.gate_up(x, ws.top, store, list(layout), bool(a4), ws.item_slot, ws.item_routes, ws.item_count, ws.h)
    ops.down(ws.h, store, list(layout), bool(a4), ws.item_slot, ws.item_routes, ws.item_count, ws.y)
    return ws.y


def combine(y, ids, weights, slots, experts, skip_negative=True, out=None):
    """out [tokens][hidden] FP32: for each token, sum = fmaf(weights[t][j], y[t * top + j], sum) over its routes in
    order, those with 0 <= id < experts (and slots[id] >= 0 when skip_negative and slots are given)."""
    tokens = ids.shape[0]
    out = torch.empty(tokens, y.shape[1], dtype=torch.float32, device=y.device) if out is None else out
    ops.combine(y, ids, weights, slots, experts, bool(skip_negative), out)
    return out


def forward(x, ids, weights, store, slots, layout, ws, a4=False, out=None):
    """The routed SwiGLU block: out [tokens][hidden] FP32, routes to negative slots skipped."""
    y = route_outputs(x, ids, store, slots, layout, ws, a4)
    return combine(y, ids, weights, slots, ws.experts, True, out)


def quantize_e2m1x2(x, global_scale):
    """crit's declared NVFP4 activation quantizer (host_int quantize_e2m1x2): x [rows][cols] FP32 -> (values int8,
    2 * e2m1; scales FP32 [rows][cols / 16], decoded / global)."""
    x = x.float().contiguous()
    values = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty(x.shape[0], x.shape[1] // 16, dtype=torch.float32, device=x.device)
    ops.quantize_e2m1x2(x, float(global_scale), values, scales)
    return values, scales
