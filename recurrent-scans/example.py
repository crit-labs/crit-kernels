# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "kernels",
#     "torch",
# ]
# ///

import torch
from kernels import get_kernel

rs = get_kernel("crit-labs/recurrent-scans", version=1)
dev = torch.device("cuda")

# A gated-delta layer of 16 key and 48 value heads (dk = dv = 128) over a 1,000-token prompt, then one decode step.
T, kh, vh, d = 1000, 16, 48, 128
q = torch.nn.functional.normalize(torch.randn(T + 1, kh, d, device=dev), dim=-1)
k = torch.nn.functional.normalize(torch.randn(T + 1, kh, d, device=dev), dim=-1)
v = torch.randn(T + 1, vh, d, device=dev)
log_decay = -torch.rand(T + 1, vh, device=dev) * 0.1   # log of the per-token decay, <= 0
beta = torch.rand(T + 1, vh, device=dev)
state = torch.zeros(vh, d, d, device=dev)               # updated in place

prompt_out = rs.gated_delta_prefill(q[:T], k[:T], v[:T], log_decay[:T], beta[:T], state, scale=d ** -0.5)
step_out = rs.gated_delta_scan(q[T:], k[T:], v[T:], log_decay[T:], beta[T:], state, scale=d ** -0.5)
print(prompt_out.shape, step_out.shape)
