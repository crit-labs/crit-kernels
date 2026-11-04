---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

Grouped expert projections for prompt chunks of mixture-of-experts models, from [crit](https://github.com/crit-labs/crit)'s
expert-streamed-prefill technique: a stable route sort by expert (no atomics), the routed rows gathered, every expert's
rows multiplied in one launch, SwiGLU, and each token's expert outputs combined in route order with its routing
weights. Each expert's weights are read through a pointer table, so an expert can sit in a device cache slot, a
staging buffer or mapped host memory.

**A baseline to replace.** It runs every format below on any NVIDIA GPU from sm_70, so a streaming schedule works
before an engine has chosen its GEMM. A provider's grouped GEMM on the format's own tensor cores (CUTLASS's ptr-array
grouped GEMM, block-scaled on sm_100 and sm_120) is usually faster: measure both on the engine's shapes and keep the
faster.

## Applicability

| | |
| --- | --- |
| Weight formats | BF16; FP8 E4M3 with one FP32 scale per row; NVFP4 (E2M1 codes, E4M3 scale per 16, a global post factor); MXFP4 (E2M1 codes, E8M0 scale per 32) |
| Activations | FP16 in, FP32 accumulation, FP16 projection outputs, FP32 combined output |
| GPUs | NVIDIA, sm_70 and newer (FP16 tensor cores through `nvcuda::wmma`); not HIP |
| Shapes | projection input width a multiple of 32, output width a multiple of 64; any token count, top-k and expert count |
| Numerics | NVFP4 and FP8 weights decode to FP16 exactly (scales applied to the accumulator); MXFP4 exactly while `code * 2^(scale - 127)` stays in FP16's range; BF16 weights round to FP16 |
| Determinism | same bits on a repeat, and a token's result independent of its chunk-mates (tests/grouped_experts_tests.cu, on an RTX 5090) |
| Accuracy | within 1e-3 of the largest output against an FP64 reference of the same decoded weights in all four formats (RTX 5090) |
| Measured speed | not measured yet |

## Check it first

Before trusting it on a target, compare `moe_block` with crit's CPU reference
(`inference/expert-streamed-prefill/code/reference.py`, `moe_block`) on a few hundred tokens of real routes and weights:
the largest difference within 2% of the largest output magnitude (FP16 tiles), and the same bits on a repeat.

## How to use
{% if functions %}

```python
# make sure `kernels` is installed: `pip install -U kernels`
from kernels import get_kernel

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
{% if source %}

## Source

The kernel-builder formatted source for this kernel is available at {{ source }}.
{% endif %}
