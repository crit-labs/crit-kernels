"""Grouped expert projections for prompt chunks, for any block weight format (BF16, FP8 E4M3 with row scales, NVFP4,
MXFP4), on any NVIDIA GPU from sm_70: a stable route sort without atomics, the routed rows gathered, every expert's
rows multiplied in one launch on FP16 tensor-core tiles with the weights decoded tile by tile, SwiGLU, and each
token's expert outputs combined in route order. A token's result does not depend on its chunk-mates.

A baseline to replace: a provider's grouped GEMM on the format's own tensor cores (CUTLASS's ptr-array grouped GEMM)
is usually faster. Check it against crit's expert-streamed-prefill reference (reference.py) on the target first."""
import struct
from typing import Sequence

import torch

from ._ops import ops

__all__ = ["FORMATS", "Workspace", "expert_table", "route_sort", "moe_block"]

FORMATS = {"bf16": 0, "fp8_e4m3_row": 1, "nvfp4": 2, "mxfp4": 3}
_MATRIX = struct.Struct("<QQf4x")   # Matrix: codes pointer, scales pointer, post (padded to 24 bytes)


def expert_table(matrices: Sequence, device=None) -> torch.Tensor:
    """One Matrix per expert from (codes, scales or None, post) tuples of CUDA tensors, as a uint8 CUDA tensor."""
    packed = b"".join(_MATRIX.pack(codes.data_ptr(), scales.data_ptr() if scales is not None else 0, float(post))
                      for codes, scales, post in matrices)
    device = device or matrices[0][0].device
    return torch.frombuffer(bytearray(packed), dtype=torch.uint8).to(device)


class Workspace:
    """The route sort's tensors for a chunk of `tokens` tokens with `top_k` routes each over `experts` experts."""
    def __init__(self, tokens: int, top_k: int, experts: int, device):
        def ints(n):
            return torch.empty(n, dtype=torch.int32, device=device)
        self.experts = experts
        self.counts, self.offsets, self.tile_offsets = ints(experts), ints(experts + 1), ints(experts + 1)
        self.row_route, self.route_row = ints(tokens * top_k), ints(tokens * top_k)

    def tensors(self):
        return self.counts, self.offsets, self.tile_offsets, self.row_route, self.route_row


def route_sort(ids: torch.Tensor, experts: int) -> Workspace:
    """Sorts the routes ids [tokens, top_k] (int32; an id outside [0, experts) is skipped) by expert."""
    ws = Workspace(ids.size(0), ids.size(1), experts, ids.device)
    ops.route_sort(ids, experts, *ws.tensors())
    return ws


def moe_block(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, fmt: str, gate_up: torch.Tensor,
              down: torch.Tensor, inter: int) -> torch.Tensor:
    """A routed SwiGLU MoE block: x [tokens, hidden] fp16, ids and weights [tokens, top_k], gate_up and down the expert
    tables (expert_table) of [2 * inter, hidden] and [hidden, inter] matrices. Returns [tokens, hidden] fp32."""
    tokens, top_k = ids.shape
    experts = gate_up.numel() // _MATRIX.size
    ws = route_sort(ids, experts)
    rows = tokens * top_k
    hidden = x.size(1)
    xp = torch.empty(rows, hidden, dtype=torch.float16, device=x.device)
    ops.gather(x, ids, experts, *ws.tensors(), xp)
    gu = torch.empty(rows, 2 * inter, dtype=torch.float16, device=x.device)
    ops.grouped_projection(FORMATS[fmt], gate_up, xp, 2 * inter, ids, experts, *ws.tensors(), gu)
    act = torch.empty(rows, inter, dtype=torch.float16, device=x.device)
    ops.swiglu(gu, act)
    yp = torch.empty(rows, hidden, dtype=torch.float16, device=x.device)
    ops.grouped_projection(FORMATS[fmt], down, act, hidden, ids, experts, *ws.tensors(), yp)
    out = torch.empty(tokens, hidden, dtype=torch.float32, device=x.device)
    ops.combine(yp, ids, weights, experts, *ws.tensors(), out)
    return out
