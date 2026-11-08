---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

Matched FP8 projections from [crit](https://github.com/crit-labs/crit)'s quantized-projections technique: E4M3 weights
(MXFP8 with its E8M0 block scales, an FP8 checkpoint's 128 x 128 FP32 block scales, or one scale per row) against E4M3
activations quantized per group, in crit's declared per-output order. Per 32-element block: one e4m3 `mma.sync`
m16n8k32 with a zero accumulator (weights the A operand, tokens B, crit's fragment order), then `lane b % 16 =
fma(block sum, weight scale * activation scale, lane)` with the product rounded, the halving tree over 16 lanes, times
post.

**Why it exists.** A decode kernel shaped for 1-8 tokens and a prompt GEMM usually sum in different orders, so a token's
logits depend on whether it was decoded, verified in a speculative round or prefilled. crit's honesty, request-history
and prompt-cache gates compare exactly that. Here `rows` (decode) and `gemm` (prompt) give the same bits for every
output, and an output depends only on its own row and token, never on the job's other tokens. Decode no longer has to
run the prompt kernel on a handful of rows to stay exact.

## Applicability

| | |
| --- | --- |
| Weight formats | E4M3 codes [rows][cols]; scales FP32 or E8M0 bytes [ceil(rows / scale_rows)][cols / scale_k]: MXFP8 (1, 32), 128 x 128 blocks (128, 128), per row (1, cols); a post factor per matrix |
| Activations | E4M3 codes per element and an FP32 scale per group (a multiple of 32); `quantize` makes them from FP32 or BF16 with crit's declared quantizer: MXFP8's 2^(floor(log2 amax) - 8) or amax / 448, round to nearest even, saturating |
| GPUs | NVIDIA sm_89 and newer (e4m3 `mma.sync`); not HIP |
| Shapes | cols a multiple of 32; decode jobs of 1 to 8 tokens on one matrix; prompt jobs of any length; any rows |
| Numerics | bitwise equal to the declared order's reference kernel on the target GPU (`tests/fp8_block_tests.cu`); within 2e-7 of FP64 relative to the products' magnitude on an RTX 5090. The tensor core's internal block sum is not specified by PTX, so the bits hold on one GPU architecture: check on yours |
| Determinism | same bits on a repeat; an output independent of its job's other tokens and of the kernel that computed it |
| Measured speed | RTX 5090: decode rows read 13.1 MB MXFP8 matrices at 0.85-1.03 TB/s for 1-8 tokens, back to back, about a plain streaming read's rate at that size (1.07-1.13 TB/s); prompt GEMM 79-97 TFLOPS on 512-4096 tokens x 5120 x 2560, without shared-memory staging yet |

## Check it first

Run `tests/fp8_block_tests.cu` on the target: the decode rows and the GEMM against the reference kernel bit for bit,
over three scale layouts and shuffled jobs, the FP64 bound, and the quantizer against its host twin. Then
`tests/fp8_block_order_tests.cu` (`nvcc -std=c++20 -O2 -arch=sm_<cc> -I tests/order tests/fp8_block_order_tests.cu`)
checks the order's assumption itself on the same GPU: three implementations of the declared order
(`tests/order/fp8_block_device.cuh`, the reference this project's kernels follow) with rows and tokens moved across
tile positions and companions, bit for bit, and the device quantizer against its host twin (`tests/order/fp8_block.h`,
a copy of crit's quantized-projections host code).

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
