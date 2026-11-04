# crit-kernels

Source for the CUDA kernels that [crit](https://github.com/crit-labs/crit) publishes to the
[Hugging Face Kernel Hub](https://huggingface.co/kernels). Each folder is one
[kernel-builder](https://github.com/huggingface/kernel-builder) project (`build.toml`, `flake.nix`, CUDA source,
tests and an applicability card, `CARD.md`) and becomes its own Hub repository, `crit-labs/<folder>`. crit's
techniques name them as `hub:crit-labs/<folder>` in their `kernels.json`.

| Kernel | Hub repository | From crit's technique |
| --- | --- | --- |
| [fp8-block-rows](fp8-block-rows/CARD.md) | `crit-labs/fp8-block-rows` | quantized-projections: matched FP8 projections (E4M3 weights and activations, block or row scales) |
| [grouped-experts](grouped-experts/CARD.md) | `crit-labs/grouped-experts` | expert-streamed-prefill: grouped mixture-of-experts projections for prompt chunks |
| [int-block-experts](int-block-experts/CARD.md) | `crit-labs/int-block-experts` | host-expert-arithmetic: 4-bit integer block expert projections, bit-identical to crit's CPU rows |
| [recurrent-scans](recurrent-scans/CARD.md) | `crit-labs/recurrent-scans` | recurrent-state-recovery: kernels for the recurrent layers of hybrid models (gated delta and more) |

Each card lists the formats, GPUs, shapes, numerics and measured speed a kernel was checked for. Read it, and run the
kernel's tests on your own target, before using it.

## Use

```python
from kernels import get_kernel

int_block_experts = get_kernel("crit-labs/int-block-experts", version=1)
```

## Build

From a kernel's folder, with Nix and kernel-builder set up:

```bash
nix run .#build-and-copy -L
```

## License

Apache-2.0, see [LICENSE](LICENSE).
