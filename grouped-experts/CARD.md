---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

Routed SwiGLU experts of mixture-of-experts models over a slot store of expert **bundles**, for decode rows and prompt
chunks alike: [crit](https://github.com/crit-labs/crit)'s grouped-experts plugin (`inference/grouped-experts`) loads
it from the kernel registry. Each expert is one contiguous bundle in crit's experts:bundles layout (its gate/up and
down matrices row major as stored, their scales and per-row factors); a call names each expert's bundle through a slot
table, so an expert can sit in the plugin's own store, a residency cache's slots or a staging ring.

A call groups its routes by expert on the device (a one-block stable sort for up to 1,024 routes, a stable device
sort beyond), runs every routed expert's gate/up rows, SwiGLU and down rows once for all of its routes on BF16 tensor
cores, and sums each token's routes in route order. No host synchronization: every grid is launched at its largest
size and exits on the device's item count, so a call at a fixed token count is captured in a CUDA graph.

**A route's bits never depend on the rows beside it**: a warp owns rows of one expert and every route of its item,
and each output is one chain of MMAs over k in ascending 64-element blocks whatever the item size, the chunk or the
other routes. Decode rows (items of 8 routes: 8 inter rows a warp, 4-bit codes decoded through a shared byte table,
operands read straight from global memory) and prompt chunks (items of 32 routes, 64 on request: operands staged in
shared memory by cp.async, the next chunk landing while the current one computes) compute the same function bit for
bit. W4A4 inputs are quantized once per route (a pre-pass) for both.

**Two arithmetics.** `tensor` (the default): BF16 tensor cores for every format. `declared`: crit's integer block order
(host-expert-arithmetic's host_int.h, which crit's CPU expert tier computes) for nvfp4 W4A4 and int4 W4A8, so a route
computed on the CPU has the device's bits: exact int32 block dots (dp4a), block b to lane b % 16, fmaf of the dot by the
rounded product of the scales, the halving tree, times the row's post; the declared SwiGLU and quantizers.

**A baseline to replace**: measure a provider's grouped GEMM on the format's own tensor cores against it on the
engine's shapes, and keep the faster one that keeps the same row invariance.

## Applicability

| | |
| --- | --- |
| Weight formats | bf16; fp8 E4M3 with one FP32 scale per row or per 128 x 128 block; nvfp4 (E2M1, E4M3 scale per 16, a 1 / global scale per row); int4 (two's complement, E4M3 scale per 16, optional 1 / global scale per row); mxfp4 (E2M1, E8M0 scale per 32) |
| Activations | BF16 (W4A16 / W8A16 / W16A16); nvfp4 W4A4: inputs and the intermediate quantized per route by crit's declared NVFP4 quantizer at each expert's input scales |
| GPUs | NVIDIA sm_80 and newer (mma.sync m16n8k16 BF16, no architecture-specific instruction; E4M3 conversion in hardware from sm_89) |
| Shapes | hidden and inter multiples of 64; any token count up to 65,535 items a call, any top-k and expert count (< 2^22) |
| Numerics | weights decoded exactly to BF16 (every format's code times its block scale is exact in BF16); FP32 MMA sums; per-row factors on the FP32 sums; fp8 block scales fmaf'd per 128-k block; crit's declared SwiGLU (IEEE operations only); the intermediate rounded to BF16 (or quantized, W4A4); the combine sum = fmaf(weight, row, sum) in route order |
| Determinism | the same bits on a repeat, alone or batched, in 8-, 32- and 64-route items, through either plan, and on graph replay (tests); the declared arithmetic equal to an exact oracle of host_int's order (tests) |
| Accuracy | within 5e-3 of the largest output of an FP64 block over the same decoded weights (3e-2 for W4A4) in every format (tests, RTX 5090) |
| Measured speed | see below |

## Interface

```python
gx = kernels.get_kernel("crit-labs/grouped-experts", version=1)
ws = gx.Workspace(tokens, top, experts, hidden, inter, device)     # the buffers of one call shape, made once
out = gx.forward(x, ids, weights, store, slots, layout, ws, a4)    # [tokens][hidden] FP32
y = gx.route_outputs(x, ids, store, slots, layout, ws, a4)        # unweighted rows [tokens * top][hidden]
y = gx.route_outputs(x, ids, store, slots, layout, ws, arithmetic="declared")   # nvfp4 W4A4 / int4 W4A8, CPU-exact
out = gx.combine(y, ids, weights, slots, experts)                  # route order; negative slots skipped
values, scales = gx.quantize_e2m1x2(x, global_scale)               # crit's declared NVFP4 quantizer
```

`layout` is `[format, bundle bytes, gate_up, gate_up_scale, gate_up_inverse, down, down_scale, down_inverse, input,
hidden, inter]` (byte offsets, -1 absent; formats 0 bf16, 1 fp8 channel, 2 fp8 block, 3 nvfp4, 4 int4, 5 mxfp4), as
crit's `bundles.kernel_layout` gives it. x is BF16 [tokens][hidden], ids int64 or int32 [tokens][top], weights FP32,
store uint8 [slots][bundle bytes] (16-byte aligned), slots int32 [experts] or None (expert e in slot e). A route whose
id is out of range or whose slot is negative is skipped (its rows of y keep what they held).

## Check it first

Run `tests/test_grouped_experts.py` on the target against the local build: every format against the FP64 block,
bitwise invariance, slot tables, capture and the quantizer.

## Measured speed

RTX 5090 (sm_120), local build (torch 2.14.1, CUDA 13.1), every expert in device memory; torch.mm BF16 8192^3 runs
207.5 TFLOPS on the same GPU. Decode: a CUDA graph of 48 layer calls, each with its own uniform random routes over one
store of every expert (so a layer's experts are mostly not in L2), time per layer call (plan, gate/up, down,
combine) and the weight bytes it moves. Prompt: one call of a chunk.

| Experts | Decode 1 / 2 / 4 / 8 tokens (us, weight GB/s) | Prompt 2,048 / 8,192 tokens (ms, TFLOPS) |
| --- | --- | --- |
| nvfp4 W4A16, 512 x top 10, 2560 x 640 | 42.2 (659) / 73.0 (753) / 131.3 (821) / 242.8 (858) | 3.84 (52.4) / 10.25 (78.6) |
| nvfp4 W4A4, same | 45.9 (605) / 76.8 (715) / 136.2 (791) / 247.9 (840) | 4.17 (48.3) / 10.60 (75.9) |
| mxfp4, same | 41.3 / 72.0 / 127.2 / 237.4 | |
| int4, same | 42.0 / 72.4 / 130.3 / 242.1 | |
| fp8 channel, 128 x top 8, 4096 x 1536 | 120.6 (1254) / 226.8 (1293) / 407.1 (1349) / 692.2 (1419) | 6.42 (96.3) / 22.16 (111.6) |
| fp8 128 x 128 block, same | 140.3 (1077) / 276.6 (1059) / 481.4 (1140) / 831.1 (1180) | 7.67 (80.6) / 23.97 (103.2) |
| bf16, same | 196.4 (1538) / 369.8 (1585) / 668.2 (1642) / 1183.2 (1658) | 6.58 (94.0) / 22.91 (108.0) |

Per-kernel (nvfp4 W4A16, 512 experts): one decode token: gate/up 27.8 us, down 9.6, plan 2.9, combine 2.2. The
small experts' gate/up pass (a k loop of 2,560 against down's 640) moves weights at about 600 GB/s against down's
about 1 TB/s: the decode limit of this baseline at that shape.
