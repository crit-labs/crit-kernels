"""Recurrent layers of hybrid models on CUDA: a short causal depthwise convolution with its carried window, gated
delta attention (Qwen3-Next / Qwen3.5 gated DeltaNet) and Mamba-2's state-space recurrence. Each has a chunked
two-pass prefill for prompts and a token-by-token scan for decode and speculative verify; the scan is column
invariant (a token decoded alone equals the same token inside a multi-token verify, bit for bit) and can write the
state after every token. All tensors are float32 on the GPU; the state is updated in place."""
from typing import Optional

import torch

from ._ops import ops

__all__ = ["causal_conv1d", "gated_delta_prefill", "gated_delta_scan", "ssd_prefill", "ssd_scan"]


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor, bias: Optional[torch.Tensor] = None,
                  silu: bool = True, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x [T, channels], weight [channels, width]; state [channels, width - 1] holds the window before row 0 and
    receives the window after the last row. Returns act(bias + sum_i weight[:, i] * x[t - width + 1 + i])."""
    out = torch.empty_like(x) if out is None else out
    ops.causal_conv1d(out, state, x, weight, bias, silu)
    return out


def _delta_out(v, out):
    return torch.empty_like(v) if out is None else out


def gated_delta_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, log_decay: torch.Tensor, beta: torch.Tensor,
                        state: torch.Tensor, scale: float = 1.0, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Chunked gated delta over a prompt. q, k [T, k_heads, dk] (normalized keys and queries), v [T, v_heads, dv],
    log_decay (<= 0) and beta [T, v_heads], state [v_heads, dk, dv] updated in place. Per token:
    delta = beta (v - e^l S^T k); S' = e^l S + k delta^T; out = scale S'^T q."""
    out = _delta_out(v, out)
    ops.gated_delta_prefill(out, state, q, k, v, log_decay, beta, scale)
    return out


def gated_delta_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, log_decay: torch.Tensor, beta: torch.Tensor,
                     state: torch.Tensor, scale: float = 1.0, checkpoints: Optional[torch.Tensor] = None,
                     out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The same recurrence token by token (decode and verify); checkpoints [T, v_heads, dk, dv] receives the state
    after every token when given."""
    out = _delta_out(v, out)
    ops.gated_delta_scan(out, state, q, k, v, log_decay, beta, scale, checkpoints)
    return out


def ssd_prefill(x: torch.Tensor, dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor, A: torch.Tensor,
                state: torch.Tensor, D: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Chunked Mamba-2 over a prompt. x [T, heads, head_dim], dt [T, heads] (after softplus and bias), B, C
    [T, groups, d_state], A [heads] (< 0), D [heads], state [heads, head_dim, d_state] updated in place.
    Per token: H' = e^{dt A} H + dt x B^T; y = H' C + D x."""
    out = torch.empty_like(x) if out is None else out
    ops.ssd_prefill(out, state, x, dt, B, C, A, D)
    return out


def ssd_scan(x: torch.Tensor, dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor, A: torch.Tensor, state: torch.Tensor,
             D: Optional[torch.Tensor] = None, checkpoints: Optional[torch.Tensor] = None,
             out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mamba-2 token by token; checkpoints [T, heads, head_dim, d_state] receives the state after every token."""
    out = torch.empty_like(x) if out is None else out
    ops.ssd_scan(out, state, x, dt, B, C, A, D, checkpoints)
    return out
