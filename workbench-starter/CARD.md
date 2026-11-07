---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

The [crit](https://github.com/crit-labs/crit) kernel workbench's starter matmul: `y[columns][N] = x[columns][K] .
w[N][K]^T` in BF16, or in E4M3 with one FP32 scale per row of `w` and per column of `x`, FP32 out. One warp per output
row and every column in that warp, FP32 accumulation in a fixed order: a column computed alone equals the same column in
a call of several.

**Why it exists.** `crit kernels new NAME --op gemv|gemm --format bf16|fp8 --starter --from <this folder>` copies it into
an engine's workbench folder as a correct starting point (allowed from a run's fourth candidate on); the engine replaces
it with the kernel it writes. It is not fast and not meant to be.

## Applicability

| | |
| --- | --- |
| Formats | BF16 x BF16; E4M3 x E4M3 with FP32 scales (x per column, w per row) |
| Shapes | x [columns <= 16][K], w [N][K], contiguous |
| GPUs | NVIDIA sm_80 and newer |
| Numerics | FP32 sums of exact products, within 1e-5 of FP64 relative to the largest output (`tests/test_workbench_starter.py`) |
| Measured speed | not measured: a starting point |

## How to use
{% if functions %}

```python
from kernels import get_kernel

kernel_module = get_kernel("{{ repo_id }}", version={{ version }})
```
{% endif %}
