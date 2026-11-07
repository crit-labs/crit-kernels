---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

CUDA kernels for the recurrent layers of hybrid language models, from [crit](https://github.com/crit-labs/crit):

- **Gated delta** (Qwen3-Next / Qwen3.5 gated DeltaNet): `gated_delta_prefill`, a chunked two-pass prefill (64-token
  chunks, register-blocked for 128-dim keys), and `gated_delta_scan`, a token-by-token scan for decode and speculative
  verify.
- **Mamba-2** state-space recurrence: `ssd_prefill` and `ssd_scan`.
- **Short causal depthwise convolution** with its carried window: `causal_conv1d`.
- **Pools of sequences** for serving many requests and speculative rounds: `pool_delta` runs the decode and verify
  rows of many sequences in one launch, each row from its own sequence's state in a pool
  `[layers][sequences][v_heads][dk][dv]` (FP32), keys and queries L2-normalized inside the scan, BF16, FP16 or FP32
  inputs, grouped key heads, dk up to 256 and any dv; a row either stashes each token's normalized key, decay and delta
  (a deferred commit: the sequence's next row replays the accepted prefix first), writes the state before and after
  every token (per-column checkpoints), keeps its start state (a recompute commit, `pool_delta_replay`), or keeps its
  final state (a prompt row). Rows come from a table the host writes or from a captured round's device tensors and a
  slot table, so a round is captured once. `pool_conv` / `conv_segments`: the causal convolution over pooled windows,
  deferred the same way, and long rows split into 32-row segments with the same bits. The commit's bookkeeping (writing
  the accepted count, selecting a checkpoint) is the caller's.

The scans are column invariant (a token decoded alone equals the same token inside a multi-token verify, bit for bit)
and can write the state after every token (`checkpoints`), which speculative decoding needs to roll back rejected
drafts. The pool rows add row invariance (a row's bits do not depend on the rows beside it) and one update expression on
every path (verify, checkpoint, stash replay, recompute), so every commit strategy gives the same bits as decoding the
accepted tokens one by one. Per-token decays are FP64 `exp` rounded once to FP32, so a decay near 1 does not drift the state over a long
decode. Everything is FP32; the state is updated in place. Key dims 64 or 128, value dims a multiple of 64 (Mamba-2:
d_state 64 or 128, head_dim a multiple of 32).

Measured on one RTX 5090: the gated-delta prefill of a 2,048-token chunk at Qwen3.8-27B's shape (16 key and 48 value
heads, dk = dv = 128) takes 1.38 ms per layer (1.32 ms measured again 2026-11-07, 5.2 ms per 8,192 tokens). At the same
shape the pool rows take 5.4 us per layer for one sequence's decode token, 12.0 us for an 8-column verify row and
56.4 us for 8 sequences x 8 columns (deferred commit; a 48-layer round of convolution, scan and commit captured as one
CUDA graph: 0.51, 1.06 and 4.28 ms); per-column checkpoints cost up to 2.5x that, the recompute commit up to 1.7x. The
tests compare every function with a float64 recurrence; tests/cuda holds the reference scans of the commit strategies
and their CUDA fixture against crit's FP64 host oracle (cmake -S tests/cuda -DCRIT_ROOT=<a crit checkout>).

## How to use
{% if functions %}

```python
# make sure `kernels` is installed: `pip install -U kernels`
from kernels import get_kernel

# If the org / user isn't a trusted publisher, pass `trust_remote_code=True` to the
# `get_kernel` call. You can find whether this kernel is from a trusted publisher
# by going to the kernel's Hub page and finding the "Trusted publisher" status at
# the top of the page.
kernel_module = get_kernel("{{ repo_id }}", version={{ version }})
{{ functions[0] }} = kernel_module.{{ functions[0] }}

{{ functions[0] }}(...)
```
{% else %}

Usage example not available.
{% endif %}

## Available functions
{% if functions %}
{% for func in functions %}
- `{{ func }}`
{% endfor %}
{% else %}

Function list not available.
{% endif %}
{% if layers %}

## Available layers
{% for layer in layers %}
- `{{ layer }}`
{% endfor %}
{% endif %}

## Benchmarks
{% if has_benchmark %}

Benchmarking script is available for this kernel. Run `kernels benchmark {{ repo_id }} --version {{ version }}`.
{% else %}

No benchmark available yet.
{% endif %}
{% if upstream %}

## Upstream

The original source code for this kernel comes from {{ upstream }}.
{% endif %}
{% if source %}

## Source

The kernel-builder formatted source for this kernel is available at {{ source }}.
{% endif %}
