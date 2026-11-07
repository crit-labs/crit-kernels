---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

A row-invariant prompt GEMM from [crit](https://github.com/crit-labs/crit)'s quantized-projections technique: `x [M][K]`
times `dense [N][K]` (both BF16 or both FP16) on tensor cores with FP32 accumulation, in Triton, in fixed tiles with one
K loop whatever `M` is: no autotuning and no split of K.

**Why it exists.** A provider's GEMM picks its kernel, and so the order of each output's sum, by the row count. A prompt
row's logits then depend on which other rows shared its GEMM, and an engine that batches the prompts of several
sequences into one pass gives each a result different from a pass of its own. Here a row's result is the same bits
whatever rows share the call, so batched prefill matches serial prefill bit for bit. crit's quantized-projections plugin
runs it for BF16 matrices and for NVFP4 ones decoded to BF16 (`prompt_gemm: fixed_tiles`) and reports the rows'
invariance in the engine's facts.

## Applicability

| | |
| --- | --- |
| Operands | `x [M][K]`, `dense [N][K]`, both BF16 or both FP16, unit column stride; any `M`, `N`, `K` |
| Output | `[M][N]` in x's dtype or the one asked for (FP32 for logits) |
| GPUs | NVIDIA with Triton's tensor-core `tl.dot` (sm_80 and newer) |
| Tiles | 128 x 128 x 64 (8 warps, 3 stages), or 64 x 64 x 64 (4 warps, 4 stages) where the large tile would not fill the device twice over; the tile picks which outputs a program computes, never an output's order |
| Numerics | FP32 accumulation of the 16-bit products, within 1e-5 of the FP64 product relative to its largest output (`tests/test_fixed_tile_gemm.py`) |
| Determinism | a row's bits independent of the call's other rows and of the tile switch; the same bits on a repeat |
| Measured speed | RTX 5090, 4,096 rows: 6-12% slower than torch's GEMM on the same operands (measured inside crit before the move) |

## Check it first

Run `tests/test_fixed_tile_gemm.py` on the target: the FP64 bound, slices of a 45,056-row call that cross the tile switch
bit for bit, and rows of a 5120-wide matrix.

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
