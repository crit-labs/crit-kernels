---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

Triton kernels for attention over paged KV caches, from [crit](https://github.com/crit-labs/crit) (its attention-state
KV pool and its attention plugins):

- **`write`**: a step's keys and values into their page slots in one launch, encoded for the page kind: BF16 or FP16,
  FP8 E4M3 with one multiplier per layer (`fp8`), FP8 E4M3 with an FP32 scale per position and KV head (`fp8:row`,
  scale = max |x| / 448, codes rounded to the nearest E4M3 value) or INT8 with an FP32 scale per position and KV head
  (`int8`, scale = max |x| / 127, codes rounded to nearest, ties to even). Slots below 0 are skipped. The codes and
  scales equal torch's encoding of the same rule bit for bit (IEEE divisions).
- **`attend`**: decode tokens and speculative-verify windows (up to 64 columns a row, causal inside the window) over the
  pages through per-row page tables, split over KV partitions: piece `i` (16 to 128 positions, set by the head width
  and the page kind) goes to partition `i mod P`, each partition walks only up to its row's last position (read on
  the device), and the partials fold in partition order (a running maximum, the sum and output rescaled to it). The
  work follows each row's length, not the cache's capacity, and one CUDA graph replays at every length. Tree masks
  (`parents`: a draft tree's columns see their ancestors and themselves), block-sparse lists (`blocks`, `block_size`:
  a token sees only its listed KV blocks), log-sum-exp outputs (`lse`), and partials packed by the rows' tokens
  (`offsets`), so a launch's workspace follows its tokens. `fused`: each program reads every partition of its columns
  in turn and folds them as the merge does, writing the output itself: the same bits as the split launch, no
  workspace, so a prompt chunk read as windows of 64 columns has the bits of its tokens decoded one by one (its value
  columns split over `vsplit` programs, each computing the scores: fewer registers, the same bits). `fresh`: each
  row's positions from a first one on read from 16-bit rows given beside the pages (a prompt chunk's own keys and
  values as the projections gave them), the earlier ones from the pages (whose bits stay as without them).
- **`prefill`**: a causal prompt chunk of one sequence over its pages (any page kind; `int8_qk` multiplies INT8 queries
  with INT8 keys on integer tensor cores); its position an int or a device tensor (one CUDA graph for every position);
  with `fresh` the chunk's own keys and values as the projections gave them (16-bit), the earlier positions from the
  pages at their kind's precision (over FP8 and INT8 pages the chunk then does not see its own quantization); a first
  fresh position (an int or a device tensor) reads the chunk's positions before it from the pages too. Another
  reduction than `attend`'s (FP32 row sums in one pass over 32- or 64-position blocks): its outputs and a decode's
  differ by the readers' rounding (up to 1.6e-2 in BF16 outputs over a few dozen positions, 2.4e-4 over 12K); a chunk
  that must have its tokens' decode bits reads through `attend(..., fused=True)` as windows of 64 columns.

Numerics: codes convert exactly to BF16 (FP16 for FP16 pages); scores and the softmax in FP32 with BF16 probabilities
on the tensor cores. A window launch's query tile (`tile_for`) holds its columns of a KV head's query heads: 16 or 32
rows in two warps, and past 32 rows 64 rows in four warps where the GPU multiplies that with mma.sync too (sm_80 to
sm_89, sm_120); on sm_90 (wgmma), sm_100 and sm_110 (tcgen05) such windows take 32-row two-warp tiles. So on each GPU
every tile is multiplied with mma.sync (the tests compile each GPU's own choice for sm_80, sm_90, sm_100, sm_110 and
sm_120 and check), and the row sums run on the tensor cores in a fixed order too. A column's result depends only on
its own keys and query, the partition count, the piece size and the page kind: not on the rows beside it, not on how
many columns or rows its launch reads (a token decoded alone equals the same token inside a verify window, bit for
bit). A launch given another tile or warp count keeps that only where the GPU multiplies it with mma.sync as well.
Errors against float64 attention over the stored values: within 8e-3 absolute (BF16 queries and probabilities),
log-sum-exps within 1e-3. E4M3 codes convert with the hardware conversion from sm_89 and with integer arithmetic before
(the same bits).

Defaults: the partition rule (`partitions`: about 1.5 programs per SM over FP8 and INT8 pages, 0.75 over 16-bit ones),
the piece size (`piece_size`: 8,192 or 16,384 head values a piece), the tiles and their warps and stages were
measured on one sm_120 GPU (an RTX 5090) and are defaults: a caller may give its own partitions and piece (both change
a column's bits, so one per cache).

Formats and shapes: head dims 16..256 (powers of two; keys and values may differ), any page size, grouped query heads
(query heads a multiple of the KV heads), any number of rows. GPUs: CUDA GPUs Triton supports (sm_80 and newer); tested
on an RTX 5090 (sm_120).

Measured on one RTX 5090 (CUDA graph replays, the GPU alone), 24 query / 4 KV heads x 256, FP8 pages with a scale
per position and head, 64-position pages, one row: 16.4 us at 4K positions and about 350 us at 200K for a decode token
(about 1,200 GB/s), 344 us for a 5-column verify window, 512 us for 8 columns; eight rows at 200K, 2.53 ms. FlashInfer
0.7.3 over FP8 pages with a layer scale on the same shapes: 283 us decode, 388 us for a 4-column window (its batch
prefill), 2.03 ms for eight decode rows.

24 query / 2 KV heads x 256, BF16 pages, one row at 200K: 290 us decode, 300 us for 5 columns, 519 us for 8
(FlashInfer: 276 us decode, 294 for 4 columns, 347 for 8). 56 query / 8 KV heads x 128, INT8 pages with a scale per
position and head: 15.3 us at 4K, 349 us at 200K, 522 us for 5 columns.

On sm_90, sm_100 and sm_110 a window of more than 32 query rows keeps the cost of 32-row tiles: it is read by several
of them, each reading the row's pages again (about the tile count times a tile's time: on sm_120 the 5-column BF16
window above took 670 us that way). The reason: their tensor cores take four warps over 64 rows as warpgroup MMA
(sm_90's wgmma, sm_100's and sm_110's tcgen05), another summation order than the mma.sync of the decode tiles, so a
64-row tile there would give a window's column other bits than the same column decoded alone; a two-warp 64-row tile
stays mma.sync but runs about 2x slower, and two 32-row dots in four warps spill. Wide windows at short contexts over
many rows are bound by the partials (FP32, value width per query head, column and partition).

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

## Benchmarks
{% if has_benchmark %}

Benchmarking script is available for this kernel. Run `kernels benchmark {{ repo_id }} --version {{ version }}`.
{% else %}

No benchmark available yet.
{% endif %}
{% if source %}

## Source

The kernel-builder formatted source for this kernel is available at {{ source }}.
{% endif %}
