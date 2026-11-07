---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

Weight-only decode GEMVs from [crit](https://github.com/crit-labs/crit)'s quantized-projections technique: 1 to 16
rows of 16-bit activations times a matrix stored as NVFP4 (W4A16), FP8 E4M3 with one scale per row or MXFP8's E8M0
scale per 32 inputs (W8A16), or BF16, on CUDA cores with FP32 sums; an NVFP4 GEMV for 1 to 8 rows in Triton; and the
decoders that write an NVFP4 matrix's values (BF16/FP16, or FP8 rows in units of a row scale) for a provider's GEMM.

**Why it exists.** No provider reachable from torch multiplies 16-bit activations by NVFP4 weights (cuBLASLt's
block-scaled GEMM takes NVFP4 on both operands), and a provider's GEMM picks its kernel, and so each output's sum,
by the row count: a decode row then changes with the rows beside it, and batched decoding or a speculative verify
window stops matching serial decoding. Here every row reduces in one fixed order whatever the row count (the rows a
warp takes are a choice of speed, never of order), so a row's result is the same bits alone and beside up to 15
others, and a 16-bit decode row reads each weight once, at about the device's streaming rate.

## Applicability

| | |
| --- | --- |
| Weight formats | NVFP4: codes [N][K / 2] (element 2i in the low nibble of byte i), E4M3 block scales [N][K / 16] row major, inverse global scale [N] FP32. FP8: E4M3 codes [N][K] with FP32 row scales [N] or E8M0 exponents [N][K / 32] (MXFP8, modelopt's layout). BF16: [N][K] |
| Activations | BF16 or FP16 [M][K], rows on 16-byte boundaries; M 1 to 16 (CUDA), 1 to 8 (Triton) |
| Output | FP32 or the activations' dtype |
| Shapes | NVFP4: K a multiple of 32 (CUDA) or 16 (Triton); FP8: K a multiple of 16 (32 for MXFP8); BF16: K a multiple of 8; any N |
| GPUs | NVIDIA sm_80 and newer; the hardware E2M1 conversion (cvt.rn.f16x2.e2m1x2) on sm_100a, 103a, 110a, 120a, 121a, the FP16 bit-pattern decode elsewhere: the same bits either way |
| Numerics | a weight's value is exact in FP32 (an E2M1 code times its E4M3 scale; an E4M3 code, times a power of two for MXFP8) and so is its product with an activation; FP32 sums in a fixed order, one xor-shuffle tree across lanes, the inverse global or row scale once at the end. Within 2e-5 of the FP64 product relative to the products' magnitude (`tests/test_decode_gemv.py`) |
| Determinism | each row's bits independent of the call's other rows, the same on a repeat; the CUDA and the Triton GEMV sum in different orders (within FP32 rounding of each other), so a process keeps to one; the Triton GEMV's tile is autotuned per shape, so its bits can differ between processes |
| Capture | no allocation, no host synchronization: the CUDA GEMVs replay from a CUDA graph |
| Measured speed | RTX 5090, NVFP4: 22-31 us for one row of a 5120 x 17408 or 34816 x 5120 matrix (about 50-100 MB, 2.3-3.2 TB/s with the L2), 434 us for a 248320 x 5120 head (715 MB, 1.65 TB/s), 597 us at 8 rows; the same bits and speed as the GEMV crit carried before (see the project's measurements in crit's migration log) |

## Check it first

Run `tests/test_decode_gemv.py` on the target: the FP64 bounds for every format at real widths, every row count alone
and batched bit for bit, both E2M1 decodes (the Triton switch and a plain-target build of the CUDA source), the
decoders' single rounding, and graph capture.

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
