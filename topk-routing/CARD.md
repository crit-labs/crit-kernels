---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

Top-k expert routing of mixture-of-experts router logits in one kernel, a warp per token row: crit's topk-router plugin
(`inference/base/topk-router`) loads it from the kernel registry for its Routing description (heterogeneous-expert-
dispatch's `routing.py`). Softmax or sigmoid scores before or after the selection, a per-expert correction bias that
only changes the selection, renormalization (divide by max(sum + epsilon, floor)) and scaling.

- **Canonical ties**: the larger value, then the lower expert id.
- **NaN-safe**: a NaN ranks below every number, so a row of NaN logits (a captured decode round's junk positions)
  still picks valid distinct experts (the lowest ids) and gets weights 0; a NaN poisons a softmax over its row as it
  does in torch.
- **One launch, no host synchronization**: a call at a fixed token count is captured in a CUDA graph.

Group limits (groups, selected_groups, group_score_top) are the plugin's torch path.

## Applicability

| | |
| --- | --- |
| Inputs | logits [tokens][experts] FP32 rows of unit stride; bias [experts] FP32 or none |
| Outputs | ids [tokens][top] int64 in selection order; weights [tokens][top] FP32 |
| GPUs | NVIDIA sm_80 and newer (warp shuffles only) |
| Shapes | experts up to 1,024, top up to 32 |
| Numerics | FP32: expf, a warp max and a warp sum for the softmax over a row; the picked weights' softmax, sum and renormalization summed in selection order |
| Determinism | the same ids and weights on a repeat and in a replayed graph; a row's picks do not depend on other rows |
| Accuracy | ids equal to an FP64 reference of the rule, weights within 2e-6 (tests, RTX 5090) |

## Interface

```python
tr = kernels.get_kernel("crit-labs/topk-routing", version=1)
ids, weights = tr.route(logits, top, bias=None, softmax=True, before=False, renormalize=False,
                        epsilon=0.0, floor=0.0, scaling=1.0, ids=None, weights=None)
```

## Check it first

Run `tests/test_topk_routing.py` on the target against the local build.

## Measured speed

RTX 5090 (sm_120), local build: see crit's notes until the publish build is measured.
