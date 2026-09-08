# GLM EWS GPU placement - 8 September 2026

Maintainer-side experiment on the same six-shard GLM-5.3-Flash UD-Q4_K_XL
artifact and RTX 4090 Laptop as the [host-cache receipt](glm53-ews-next-20260908.md).
This is a hybrid GPU placement, not all-GPU inference or independent replication.

## What changed

The experimental runtime patch and DLLs are unchanged. The serving profile
changes only `cpu_moe: true` to `false`: eight slots, `gpu_layers: 20`, eight CPU
threads, F16 KV, flash attention off, MTP off, TF32 off, CUDA graphs off.
The existing slab installer can target host or CUDA buffers. This lot adds
bound host/device cache allocation counters and GPU numerical test profiles;
it does not add an execution-permission layer.

The runtime counts 47 placement positions: 46 blocks (including unused MTP)
plus output. `gpu_layers: 20` assigns positions 27..46 to CUDA, including
**18 routed trunk layers (27..44)**. The other **24 routed layers (3..26)**
remain on CPU. `gpu_layers` is not the number of active routed layers.

| Routed expert tensor allocation, 8 slots | Bytes |
|---|---:|
| CUDA device cache | 2,198,339,584 |
| Host cache | 2,953,838,592 |
| Total physical cache | 5,152,178,176 |
| Logical expert tensors in the GGUF | 185,478,414,336 |

Other model weights, recurrent/KV state, scratch, driver memory and Next's
resident/embedding models are additional. Tensor bytes are not process peaks.

## Numerical checks and the control's boundary

- **Native vs EWS:** `gpu_layers: 4` fits native expert weights on this GPU.
  It places routed layers 43/44 on CUDA, alongside output; MTP remains unused.
  The six-token English prompt produces eight predicted tokens, and all
  **1,239,040 compared float logits are bit-identical**, maximum delta 0.
  EWS installs 2,426,142,720 cumulative device payload bytes in a
  252,182,528-byte device expert cache. This is a real CUDA-expert control,
  but it covers only those two routed layers.
- **Larger placement:** native full expert weights at `gpu_layers: 20` do not
  fit this GPU. Instead, compare 16-slot and 8-slot EWS caches at that same
  placement. English, Python completion and French completion all match
  bit-for-bit: 1,239,040 logits and eight generated tokens per comparison.
  Prompt lengths are 6, 19, 21 respectively. These are three short prompts,
  **not millions of independent observations**.
- The second check is **cache-size consistency, not native equivalence**.
  Both caches can share a defect. CPU/GPU bit-identity is not asserted either:
  comparisons hold arithmetic placement fixed.
- **Retained initial probe:** `gpu_layers: 1` passed numerically but placed
  only output on CUDA. Device expert bytes and payload were zero, so it was
  not accepted as GPU-expert evidence. The native control was changed to 4.
- Updated-reader Gemma regressions on the original pinned runtime pass all
  four French/code comparisons (8 and 32 slots vs native, eight predictions).
  Unit checks pass 46 assertions per runtime; offline serving passes 628.
  Full EIE also builds and links against the unchanged original runtime.

Exact prompts, build/configuration commands and interpretation are in the
[GPU reproduction section](../GLM_EWS_EXPERIMENT.md#gpu-placement-reproduction).
Do not run these standalone placement tests alongside a production resident.

## Actual Next cohabitation

The first live test exercises Next's real session, tools and 12B resident,
with new empty state, Nomic, a configured 16k resident context and 23 exposed
tools. It is text-only, with idle drivers off; configured capacity is not a
filled 16k prompt. Production state is not copied or modified.

The resident answers 13, calls GLM for a counter-examination of correlated
retry failures, incorporates the response, then answers 55. **That loop works,
but the complete-answer gate fails in live-v1**: GLM reaches the 768-token
ceiling, `finish_reason=length`, `incomplete=true`, after 1,103.031 s.
All three resident turns complete, and all experiment-owned processes stop.
Original Next state is unchanged.

The raw auxiliary answer is not a quality success: it exceeds two sentences,
contains malformed mathematics and overgeneralizes dependence as perfect
correlation. The resident's synthesis repeats the latter simplification.
These outputs are retained, not polished into a better benchmark answer.

Live-v1 records 35,826 callbacks, 78,594 hits, 208,014 misses;
1,261,077,069,824 bytes explicitly installed in device expert buffers and
1,928,837,201,920 in host buffers. Reader bytes total 3,192,470,347,776 including
repeated reads/alignment. These are cumulative copies, not distinct model size.
GPU use peaks at a sampled 14,995 MiB, with approximately 1,054 MiB free.

A new fresh-state run uses a 1,536-token auxiliary allowance; **live-v2 passes
the complete-answer and resident-continuation gates**. Runtime, placement and
user request are unchanged, but
the resident regenerates its tool question (omitting the final instruction
to explain). Therefore it is a new functional trial, not a controlled
budget-only ablation or paired speed comparison.

| Live-v2 | Measurement |
|---|---:|
| Complete GLM answer | 148 tokens, `stop`, `incomplete=false` |
| Auxiliary wall time | 278.515 s |
| Resident before / after | 13 / 55, with actual tool call and synthesis between |
| EWS callbacks / hits / misses | 9,282 / 21,736 / 52,520 |
| Explicit device expert payload | 313,628,819,456 bytes |
| Explicit host expert payload | 491,761,893,376 bytes |
| Reader bytes, repeated reads/alignment included | 806,036,078,592 bytes |
| Sampled total GPU peak / minimum free | 14,995 / 1,054 MiB |
| Sampled EIE working-set peak | 9,641,463,808 bytes |
| Sampled EIE private committed memory peak | 20,000,854,016 bytes, not physical RAM |
| Minimum sampled available whole-host RAM | 1,116,454,912 bytes |

There are 32 samples at approximately ten-second intervals. Whole-host samples
include brief CPU-only contract-test builds; this is not an isolated RAM
benchmark. After cleanup the GPU returns to 589 MiB, the three test listeners
are closed, and original Next state has the same digest as before the run.

The second response gives the persistent-outage counterexample in two body
sentences after a short introduction. It is not a full characterization of
all dependent failures. The resident incorporates it and continues normally.
Since the successful answer is only 148 tokens, it would fit the earlier
768-token ceiling too: **success cannot be attributed to the larger allowance**.
The different question and new session must not be hidden in a speedup claim.

The [public receipt](data/glm53-ews-gpu-next-20260908.json) includes unedited
generic questions/answers, both live attempts, counters and source/model/binary
hashes. Private Next context envelopes are excluded; replaying that integration
requires Next. The standalone numerical and EIE HTTP recipes do not.
Receipt SHA-256 (UTF-8/LF):
`e93e184a1a96dddd4019e2b2309381804da103d102b5c77bd2f777f42b475015`.
Next uses standalone endpoint provenance, not a Job-owned launch attestation;
the experiment separately records its actual owned EIE process and binaries.

## Boundaries

No all-GPU-expert, long-context, MTP, multimodal, multi-GPU, general quality,
energy or training-GPU claim is established. Timings have no declared
cold-cache protocol and are not matched speed comparisons with earlier runs.
Explicit reader/copy bytes are not all SSD-controller or PCIe traffic.
Production Next, native GLM reference and EIE's default runtime pin are unchanged.
