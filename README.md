# crit-kernels

Source for the CUDA kernels that [crit](https://github.com/crit-labs/crit) publishes to the
[Hugging Face Kernel Hub](https://huggingface.co/kernels). Each folder is one
[kernel-builder](https://github.com/huggingface/kernel-builder) project (`build.toml`, `flake.nix`, CUDA source,
tests and an applicability card, `CARD.md`) and becomes its own Hub repository, `crit-labs/<folder>`. crit's
techniques name them as `hub:crit-labs/<folder>` in their `kernels.json`.

| Kernel | Hub repository | From crit's technique |
| --- | --- | --- |
| [fp8-block-rows](fp8-block-rows/CARD.md) | `crit-labs/fp8-block-rows` | quantized-projections: matched FP8 projections (E4M3 weights and activations, block or row scales) |
| [fused-norms](fused-norms/CARD.md) | `crit-labs/fused-norms` | fused-norms: RMSNorm with a residual, head norms with rotary positions, gated norms, activation products and residual-stream mixes, one Triton launch per call (a noarch kernel: Python and Triton, no CUDA build) |
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
kernel-builder build-and-copy --max-jobs 2 --cores 6 -L .
```

This builds every torch and CUDA variant for every architecture into `build/`. Without the two options Nix may build
as many variants at once as there are CPU cores, each using every core, which can exhaust a workstation's memory:
`--max-jobs` (variant builds at once) times `--cores` (cores per variant build) bounds the compiler processes.

## Local kernel development

Before a kernel is published, test it from a build for your own GPU and torch only (one or two minutes, against tens of
minutes for the full build), loaded by `kernels` in place of the Hub repository:

1. The build goes to `<kernel>/build/<variant>/`: the Python package, the compiled extension and `metadata.json`, the
   layout kernel-builder writes. `kernels` picks the variant matching the running system:
   `torch<major><minor>-cu<CUDA>-<cpu>-linux` for a CUDA kernel, where CUDA is the runtime torch was built with
   (`torch.version.cuda`; a variant with a higher CUDA minor version is refused, so a build compiled with a newer
   toolkit than torch's is named after torch's), `torch<major><minor>-cpu-<cpu>-linux` for a CPU kernel (loaded with
   `get_kernel(..., backend="cpu")`), and `torch-<backend>` for a Python-only (`[torch-noarch]`, e.g. Triton) kernel.
2. To build it: copy the folder, list in each CUDA kernel's `cuda-capabilities` in the copy's `build.toml` only what
   your GPU runs (the generated CMake otherwise compiles every listed capability), `kernel-builder create-pyproject -f
   <copy>`; `cmake -S <copy> -B <dir> -G Ninja -DPython3_EXECUTABLE=$(which python) -DBUILD_ALL_SUPPORTED_ARCHS=ON
   -DCMAKE_INSTALL_PREFIX=<stage>` (add `-DGPU_LANG=CPU` for a CPU kernel), `cmake --build <dir> -j 4` and
   `cmake --install <dir>`; move the variant folder from `<stage>` into `<kernel>/build/` under the name above and
   record its file digest with `kernel-builder hash <kernel>`. A Python-only kernel needs no compiler: its
   `torch-ext/<package>/` folder, the `_ops.py` create-pyproject writes and `metadata-<backend>.json` (as
   `metadata.json`) make the variant.
3. `export LOCAL_KERNELS=crit-labs/<kernel>=/path/to/<kernel>` (several entries separated by `:`). Then
   `get_kernel("crit-labs/<kernel>", version=1)`, and crit's registry kernels, load the local build. A repository named
   in `LOCAL_KERNELS` is not looked up on the Hub, so name only kernels that have a build.
4. Run the kernel's tests on it: `python -m pytest <kernel>/tests`.

## License

Apache-2.0, see [LICENSE](LICENSE).
