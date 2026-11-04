---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

Integer block expert projections from [crit](https://github.com/crit-labs/crit)'s host-expert-arithmetic technique:
4-bit weights (NVFP4's E2M1 codes or INT4, one E4M3 scale per 16) against activations quantized per block (E2M1 x 2
for 4-bit activations, INT8 for 8-bit ones), in crit's declared integer order. Each 16-block's dot of the integer values
is exact on any processor; the FP32 step is declared (block b to lane b % 16, `lane = fma(sum, weight scale *
activation scale, lane)` with the product rounded, the halving tree over 16 lanes, times post).

**Why it exists.** The decode rows, the prompt GEMM and crit's CPU rows (`host_int.h`: AVX-512 VNNI, AVX2, scalar)
give the same bits for every output. A CPU expert tier, a device expert cache and the prompt path can split an expert's
work any way, and move experts between tiers while serving, without changing a token: what crit's repeatability,
request-history and honesty gates ask, at integer speed on every processor.

## Applicability

| | |
| --- | --- |
| Weight formats | E2M1 codes (NVFP4) or INT4, two a byte (low nibble the even element), E4M3 scale per 16 weights, a post factor per matrix; rows layout [rows][cols / 2] |
| Activations | int8 values per element (E2M1 x 2 or INT8) and an FP32 scale per 16 or 32 values; the declared quantizers make them from FP32 |
| GPUs | NVIDIA sm_80 and newer (int8 `mma.sync` m16n8k16, `dp4a`); not HIP |
| Shapes | cols a multiple of 64; decode jobs of 1 to 8 tokens on one matrix; prompt jobs of any length; any rows |
| Numerics | bitwise equal to the declared order's scalar oracle and to crit's CPU rows (tests/int_block_tests.cu, an RTX 5090) |
| Determinism | same bits on a repeat; an output independent of its job's other tokens and of the kernel that computed it |
| Measured speed | RTX 5090: decode rows 1.9-3.4 µs per 2.76 MB matrix for 1-8 tokens (from L2); prompt GEMM 24 TOPS on 4096 x 1920 x 2560, a first version without shared-memory staging |

## Check it first

Run `tests/int_block_tests.cu` on the target: the rows and the GEMM against the scalar oracle, bit for bit, and the
quantizers and SwiGLU against their declared host functions. An engine that splits experts between the CPU and the GPU
checks crit's `host_int_tests` too, so its CPU rows match on the target CPU.

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
