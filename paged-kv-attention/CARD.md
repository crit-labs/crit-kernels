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
  the device), and the partials merge in partition order by their log-sum-exps. The work follows each row's length,
  not the cache's capacity, and one CUDA graph replays at every length. Tree masks (`parents`: a draft tree's columns see their ancestors and themselves), block-sparse
  lists (`blocks`, `block_size`: a token sees only its listed KV blocks) and log-sum-exp outputs (`lse`).
- **`prefill`**: a causal prompt chunk of one sequence over its pages (any page kind; `int8_qk` multiplies INT8 queries
  with INT8 keys on integer tensor cores).

Numerics: codes convert exactly to BF16 (FP16 for FP16 pages); scores and the softmax in FP32 with BF16 probabilities
on the tensor cores; the row sums run on the tensor cores too, so a column's bits do not depend on the tile shape. A
column's result depends only on its own keys and query, the partition count and the page kind: not on the rows beside
it, not on how many columns its launch reads (a token decoded alone equals the same token inside a verify window, bit
for bit). Errors against float64 attention over the stored values: within 8e-3 absolute (BF16 queries and
probabilities), log-sum-exps within 1e-3. E4M3 codes convert with the hardware conversion from sm_89 and with integer
arithmetic before (the same bits).

Formats and shapes: head dims 16..256 (powers of two; keys and values may differ), any page size, grouped query heads
(query heads a multiple of the KV heads), any number of rows. GPUs: CUDA GPUs Triton supports (sm_80 and newer); tested
on an RTX 5090 (sm_120).

Measured on one RTX 5090 (CUDA graph replays, the GPU alone), 24 query / 4 KV heads x 256 (Qwen3.8-27B's attention),
FP8 pages with a scale per position and head, 64-position pages, one row: 17.8 us at 4K positions, 57.8 us at 32K,
344 us at 200K for a decode token (1,210 GB/s), 331 us for a 4-column verify window (1,256 GB/s), 492 us for 8
columns; eight rows at 200K, 2.46 ms (1,352 GB/s). FlashInfer 0.7.3 over FP8 pages with a layer scale on the same
shapes: 283 us decode, 388 us for a 4-column window (its batch prefill), 2.03 ms for eight decode rows.

24 query / 2 KV heads x 256, BF16 pages, one row at 200K: 275 us decode (1,488 GB/s), 280 us for 4 columns, 478 us
for 8 (FlashInfer: 276, 294, 347 us); FP8 pages with a scale per position and head: 168 us decode (1,239 GB/s).
56 query / 8 KV heads x 128, INT8 pages with a scale per position and head: 14.5 us at 4K, 337 us at 200K (1,254
GB/s), 342 us for 4 columns; BF16 pages at 200K: 511 us (FlashInfer: 531 us).

Wide windows at short contexts over many rows are bound by the partials (FP32, value width per query head, column and
partition): 8 rows of 8 columns at 4K positions write more partials than they read pages.

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
