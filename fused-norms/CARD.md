---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

The elementwise work around a transformer's matmuls, one Triton kernel per call, from
[crit](https://github.com/crit-labs/crit)'s fused-norms technique (its `norms` slot): RMSNorm with a residual added in
place (an FP32 residual stream beside 16-bit rows), an offset weight (a zero-centred weight stored as weight - 1) and a
chosen output dtype; query/key head norms, alone or with rotary positions; the gated RMSNorm of linear-attention
outputs; rotary positions (half or interleaved pairs, partial rotation, inverse frequencies or a cos|sin table); SiLU,
GELU and sigmoid products; L2 norms; and the mixes of several residual streams: a gated, weighted or plain sum over
the streams and one row added into every stream with a weight per stream.

**Why it exists.** Written in torch, each of these is a chain of small kernels (a cast, a square, a mean, an rsqrt,
multiplies, a cast back): a decode round launches hundreds of kernels that each move a few kilobytes, and a prompt pass
reads the same activations several times. Here each call is one launch that reads its inputs once. And torch's own
mean over a row does not keep a row's bits across row counts (a row of a few-row call sums in another order than in a
long call), which an engine whose decode, verify and prompt rows must agree cannot use; every kernel here reduces a
row in an order fixed by its width alone.

## Applicability

| | |
| --- | --- |
| Functions | `rms_norm`, `rms_norm_heads`, `rms_norm_rope`, `gated_rms_norm`, `rope`, `silu_mul`, `gelu_mul`, `sigmoid_mul`, `l2_norm`, `stream_sum`, `stream_add`; signatures and semantics in the package docstrings. Every norm weight may be one row per group (`[groups][dim]` for x `[..., groups, dim]`: a weight per head or per residual stream) |
| Dtypes | BF16, FP16, FP32 inputs, each read in its own dtype (a residual or gate may differ from the rows); FP32 inside; one rounding to `out_dtype` (BF16, FP16, FP32) |
| Layouts | the last dimension contiguous, the other dimensions any strides (a query beside its gate in one projection output, a slice of a fused projection); layouts whose leading dimensions do not merge into two levels are copied first |
| GPUs | NVIDIA GPUs Triton compiles for (sm_80 and newer; checked on sm_120) |
| Shapes | any row count; rows up to 16,384 values in one pass, wider rows (norms and gates) in two passes of 4,096; rope heads up to 16,384 values |
| Numerics | within FP32 rounding of a float64 reference with FP32 out, within one output rounding with 16-bit out (`tests/test_fused_norms.py`); the rotary angle is `fp32(position) * fp32(inverse frequency)`, as checkpoints' reference code computes it; with a residual, the sum as stored in the residual's dtype is what is normalized |
| Determinism | same bits on a repeat; a row's bits independent of the rows beside it and of the row count (1 to 8 rows and 320 checked at four widths); CUDA graph replay equal to eager calls |
| Capture | no host synchronization, nothing allocated but the result (or `out=`); a shape's first call compiles, so call each shape once before capturing a graph |
| Measured speed | PENDING |

## Check it first

Run `tests/test_fused_norms.py` on the target: every function against float64, rows alone and batched, strided views
against their copies, `out=` and in-place rope, and a captured graph against eager calls.

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
