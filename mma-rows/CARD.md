---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

Projections of 1 to 64 activation rows on tensor cores in one fixed tile, from
[crit](https://github.com/crit-labs/crit)'s quantized-projections technique: a matrix stored as NVFP4, FP8 (E4M3 with
a row scale), MXFP8 or BF16 times BF16 rows, the activations used as they are (weight only: W4A16, W8A16) or quantized
as a prompt's are (FP8 per row: W8A8; NVFP4 per 16-value block: W4A4), FP32 sums; and the two activation quantizers
alone, for a provider's GEMM on the same activations.

**Why it exists.** Decode steps, speculative verify windows and short prompts multiply a few rows by every weight: the
weights' read is the cost. A CUDA-core GEMV grows compute bound past a few rows, and a provider's GEMM picks its
kernel, and so each output's sum, by the row count, so a row changes with the rows beside it. Here every call runs one
fixed MMA tile (8 rows a column of the MMA, the weights its A operand), each weight is read once a call whatever the
rows, and a sum's order depends on the matrix and the plan only: a row gets the same bits alone, in a decode call of
16 rows or in a prompt's chunk of 64, which keeps batched decoding, verify windows and padded prompt buckets equal to
serial decoding.

## Applicability

| | |
| --- | --- |
| Weight formats | NVFP4: codes [N][K / 2] (element 2i in the low nibble of byte i), E4M3 block scales [N][K / 16] row major or in cuBLASLt's 128 x 4 tiled layout, inverse global scale [N] FP32. FP8: E4M3 [N][K] with FP32 row scales [N]. MXFP8: E4M3 [N][K] with E8M0 exponents [N][K / 32] (modelopt's layout). BF16: [N][K] |
| Activations | BF16 [M][K], M 1 to 64, rows on 16-byte boundaries; used as they are, or quantized by the kernel's pre-pass: FP8 rows (scale = amax * fp32(1 / 448), at least 1e-12; values e4m3(clamp(x / scale)), the scale after the sum) or NVFP4 blocks under the input scale g (block scale e4m3(min(amax * fp32(1 / 6) * g, 448)), values the nearest E2M1 magnitude, ties to the even code; fp32(1 / g) after the sum) |
| Output | FP32, BF16 or FP16, rounded once |
| Shapes | K a multiple of 32, any N |
| GPUs | NVIDIA sm_80 and newer (mma.m16n8k16 with BF16 operands); the hardware E2M1 conversion on sm_100a, 103a, 110a, 120a, 121a, the bit-pattern decode elsewhere (the same values); the pre-pass overlaps the weight stream through programmatic dependent launch from sm_90 |
| Numerics | every value enters the MMA exactly in BF16 (an E2M1 code times its E4M3 scale, an E4M3 code, times its power of two for MXFP8, a BF16 weight; a BF16 activation or its quantized value), FP32 accumulation, the row, activation and global scales after the sum. Within 2e-5 of the FP64 product relative to the products' magnitude (`tests/test_mma_rows.py`) |
| Determinism | a sum's order is fixed by N, K and the plan's splits of K (plan(): from the shape and the device's multiprocessor count); the rows a call takes, `groups` and `warps` change no bits. The split CTAs' partial sums add in split order (the last CTA of a row tile adds them). A row's bits are the same alone and among up to 63 others, and on a repeat |
| Capture | no allocation and no host synchronization (the caller gives scratch, partial-sum and counter buffers; the counters are left zero): a CUDA graph captures a call |
| Measured speed | RTX 5090 (see the tables in the measurements below) |

## Check it first

Run `tests/test_mma_rows.py` on the target: the FP64 bounds for every format with and without splits of K, every row
count 1 to 64 alone and batched bit for bit whatever the groups and warps, the quantizers' bits against the torch
quantizers they follow, and graph capture with the counters left zero.

The NVFP4 decode follows a crit-labs engine's tensor-core NVFP4 GEMV (Apache-2.0), generalized to every format, split
K and the quantized activations.

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
