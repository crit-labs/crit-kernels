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

The scans are column invariant (a token decoded alone equals the same token inside a multi-token verify, bit for bit)
and can write the state after every token (`checkpoints`), which speculative decoding needs to roll back rejected
drafts. Per-token decays are FP64 `exp` rounded once to FP32, so a decay near 1 does not drift the state over a long
decode. Everything is FP32; the state is updated in place. Key dims 64 or 128, value dims a multiple of 64 (Mamba-2:
d_state 64 or 128, head_dim a multiple of 32).

Measured on one RTX 5090: the gated-delta prefill of a 2,048-token chunk at Qwen3.8-27B's shape (16 key and 48 value
heads, dk = dv = 128) takes 1.38 ms per layer. The tests compare every function with a float64 recurrence.

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
