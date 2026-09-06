# Consumed EWS: measured results and boundaries

Campaigns: 5 September 2026. Evidence rechecked locally: 7 September.
These are maintainer-side executed experiments, not independent replication.
No inference was rerun for publication.

## Three stages, not one interchangeable benchmark

| Stage | What ran | What it establishes |
|---|---|---|
| Standalone consumed pilot | Patched llama.cpp pilot, Gemma 26B, reference and bounded expert slots | Real streamed-weight consumption, counters and numerical comparison |
| EIE target numerical port | Pinned EIE llama.cpp fork plus supplied patch, two prompts and 32 predictions | Bit identity on four candidate/reference comparisons |
| Real Next envelope | Actual Next on copied state, resident 12B plus separate streamed 26B | Bounded coexistence, observed context, tool calls, memory and request timings |

The older [August C+ experiment](ews-gemma4-a4b-rtx4090-laptop.md)
inserted real I/O while FFNs still used resident weights. Its figures must not
be substituted for any stage above.

## Artifact identity and machine

- Windows, RTX 4090 Laptop 16 GB, driver 595.79, i9-14900HX, 32 GB host RAM.
- Pilot runtime base: `fae3a28070fe4026f87bd6a544aba1b2d1896566`, with local consumed-weight patch.
- EIE target runtime: `2168b0cd8b87c75c29a1e6588692ebbb805b9bd2`, with
  [published patch](../../patches/ews-runtime-2168b0.patch).
- 26B GGUF: 14,439,363,584 bytes, SHA-256
  `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`.
- Resident 12B GGUF: 6,975,879,008 bytes, SHA-256
  `f568ac5de71c8fcac5d5794494388ad94db9e18b4368ca897e21b30d2448eeec`.

Full profiles, binary hashes and per-case counters:
[pilot JSON](data/ews-consumed-pilot-20260905.json),
[runtime/envelope JSON](data/ews-runtime-integration-20260905.json).

## The weights were transferred and consumed

The pilot's physical expert buffers scale with the per-layer slots, rather
than allocating the full expert set and merely simulating eviction. The weight
matmuls use physical slot IDs; router/scaling metadata retain logical IDs.

| Slots/layer | Physical expert bytes | CUDA weight buffers, decimal GB |
|---|---:|---:|
| Full reference (128 experts) | 12,846,366,720 | 14.423603 |
| 32 | 3,211,591,680 | 4.788828 |
| 16 | 1,605,795,840 | 3.183032 |
| 8 | 802,897,920 | 2.380134 |

These are **weight buffers**, not total GPU usage. A 605,552,640-byte CPU
weight buffer, KV, activations, runtime and driver allocations are additional.

For the French 32-slot candidate over 95 decode calls:

- 22,800 expert accesses, 18,324 hits and 4,476 misses.
- **14,974,046,208 payload bytes transferred to GPU.**
- **15,010,709,504 bytes read**, larger because direct reads are aligned.
- Actual eviction occurs; the 8-slot case also forces substantially more misses.

These are instrumented application-byte counters, not an external PCIe bus
measurement or proof that every file read reached physical NAND. There is no
full expert RAM cache; the reported aligned read buffer is 2,236,416 bytes.

## Numerical gate and negative control

Pilot: two prompts (French 75 tokens, code 69 tokens), 512-token context,
one-token microbatches, 96 predictions, 95 decode calls, greedy generation.
References and candidates use matched callback/graph settings; CUDA graphs
are disabled. This is not a fast-default-reference throughput comparison.

- **Five positive comparisons:** each compares 25,165,824 FP32 logit values;
  byte-identical logits, generated IDs and routes; maximum absolute difference 0.
- **One negative control:** zeroed expert weights, teacher-forced reference
  inputs; logits diverge, maximum absolute difference **44.680049896**.
  Its speed is not counted as useful-model performance.
- Reference/candidate logit SHA-256:
  - French: `898e141a9a55d1bdbf3d65209a89fcb1c172102b1df323aa148d558ad04256b4`.
  - Code: `5b53d01b4e7c542abc96de89a5c52f0a8706931a35ff636cc6ab2fc16be9bd1b`.

The separate EIE target gate uses F16, context 512 and 32 predictions:
four comparisons (French/code at 32 and 8 slots), **33,554,432 FP32 values
across comparisons**, maximum absolute difference 0, identical generated IDs.
This is finite-profile evidence, not a proof for arbitrary text or hardware.

## Pilot speed: instrumented, single-pass samples

| Case | Prefill seconds | Decode tokens/s |
|---|---:|---:|
| French full reference | 1.292 | 65.04 |
| French 32 slots | 6.302 | 12.40 |
| French 8 slots | 15.857 | 4.76 |
| Code full reference | 1.389 | 56.94 |
| Code 32 slots | 6.041 | 11.00 |
| Code 16 slots | 10.839 | 6.75 |
| French 32 slots, 12B loaded | 6.427 | 11.04 |

The tradeoff is visible: lower physical weight residency costs latency.
There are no repeat distributions or thermal controls here. These figures
are not a promise of a fixed serving rate.

The pilot's resident 12B had only a 256-token context and a small warm-up.
Its post-run GPU snapshot was 13,490,413,568 bytes used, not a measured peak.
It remained usable after the 26B run; this alone is not simultaneous generation.

## Real Next, not just an empty resident

The target envelope ran two five-turn sessions on copied Next state:

- Resident 12B context configured at 16,384; observed prompts up to **9,494 tokens**.
- Streamed 26B: **16 slots**, context 2,048, F16, separate EIE process.
- Four auxiliary 26B requests ran alongside Next. Each produced 192 tokens in
  **32.17-33.34 seconds**, or **5.76-5.97 output tokens per request-wall-second**.
  This includes prefill and contention, not isolated decode.
- Externally sampled peak total GPU use: **13,408 MiB**; minimum free: **2,641 MiB**.
- State copy covered 1,528 files; original-state before/after checks matched.

Keep the failed case: in the first envelope, the model did not call tools and
invented file claims. The second, explicitly tool-prompted envelope recorded
actual runtime, memory, belief-search and file-list tool calls.

Limitations: no actual vision inference, no observed consolidation/reflection
event in these short runs, no completed belief revision, and no general answer
quality assessment. Nomic and vision services being alive is not equivalent to
exercising all their functions. This stage is coexistence; later automatic
Next-to-26B delegation is a separate campaign.

## Publication and reproduction

Published: metadata, hashes, two prompts, runtime patch/cache code,
[numerical runner](../../experiments/ews_target/run_forward.py),
[forward harness](../../tests/ews_forward.cpp) and optional Next envelope script.
The latter needs a separate Next checkout and copied state; Next is not bundled.

Raw logit tensors and private Next reports are **not** included. Their existing
local bytes were hashed again: 24 pilot artifact files, six runtime logit files
and two Next reports matched. The complete old standalone pilot implementation
is not bundled; the published target runner is the current reproduction route.
A third party can run that route with the named model/toolchain, but cannot
recompute every historic number from the JSON summaries alone.

For offline metadata checks, Python 3.9+:

```bash
python tests/claims/verify_consumed.py
```

Without raw directories, the script explicitly reports raw evidence as
"not supplied"; that is not a successful raw replication. See the
[verification receipt](claims-verification-20260907.md) and
[runtime build guide](../ews/runtime-port.md).
