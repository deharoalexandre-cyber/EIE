# Public claims audit - 7 September 2026, serving update 8 September

## Scope and status

Requested by Alexandre De Haro; performed with Codex through source inspection,
offline diagnostic probes and re-verification of existing local evidence.
This is a maintainer-side audit, **not independent replication, certification,
or a new GPU campaign**.

The initial public revision was `07b1500e6b5f68b5fb12a3118aff832d217bbfc6`.
It lagged the locally tested EWS work based on `67a779d`. This publication
imports that work, retaining the intervening Apple Silicon build changes.
The measured binaries are identified in the evidence; they are not asserted
to be byte-identical to a fresh build of this reconciled source tree.

Scope: this repository's README, About description, desktop/mobile guides,
benchmark reports, configuration examples and implementation. Linked model
cards, publications, websites, upstream repositories and old Git revisions
are not retroactively certified or rewritten. Frozen research artifacts remain
unaltered and have a scope notice in their directory.

Labels:
- **Implemented:** an inspected code path, not a reliability guarantee.
- **Locally verified:** an identified bounded executed test and retained evidence.
- **Maintainer-reported:** reported measurements without a complete raw bundle inspected here.
- **Incomplete / unqualified:** missing behavior or missing evidence; not a current guarantee.

See the [verification receipt](benchmarks/claims-verification-20260907.md)
and [acceptance roadmap](ROADMAP_TO_CLAIMS.md). The [8 September serving receipt](benchmarks/serving-functional-20260908.md)
updates the affected code findings below, without reusing old GPU results as
qualification of the new candidate.

## What the publication adds, not what remains to invent

| Claim | Finding | Public evidence |
|---|---|---|
| Streamed experts actually participate in the forward pass | **Locally verified.** The September slot-index path feeds loaded expert slabs to weight matmuls; zeroing the weights changes logits in the pilot | [Report](benchmarks/ews-consumed-20260905.md), [reader/cache](../backends/expert_stream.cpp), [runtime patch](../patches/ews-runtime-2168b0.patch) |
| Expert traffic and physical weight savings were measured | **Yes.** FR/32-slot pilot: 14,974,046,208 decode payload bytes; CUDA weight buffers 14.424 to 4.789 decimal GB | [Pilot metadata](benchmarks/data/ews-consumed-pilot-20260905.json); not whole-process peak VRAM or PCIe bus-analyzer traffic |
| Streaming preserves reference logits | Five positive pilot and four runtime comparisons are bit-identical within their exact profiles | Two short prompts; no claim of all-input/all-quantization equivalence |
| Resident Next and streamed 26B coexist | Two five-turn copied-state runs; real 12B, memory and tools; four auxiliary HTTP requests in total, alongside Next activity | [Runtime metadata](benchmarks/data/ews-runtime-integration-20260905.json); not proof of every lifecycle function |
| Correct TurboQuant KV type mapping | Imported correction maps KV names to `GGML_TYPE_TURBO*_0`, not similarly named weight formats | [Backend](../backends/cpu_backend.cpp); F16 numerical gate and separate Turbo3 API smoke, not a TurboQuant quality study |
| Device memory counters are real in an inference build | Imported backend uses `ggml_backend_dev_memory`; no-llama CUDA/HIP placeholders still return constants | Same backend, [admin endpoints](../server/api.cpp); policies remain unenforced |
| Auxiliary request failure semantics and prefill cancellation | Existing local fixes imported: strict model, overflow option, length finish reason, propagated errors, disconnect checks | Same backend/API; [receipt](benchmarks/claims-verification-20260907.md) records bounded cancellation results |

The September consumed runtime uses a bounded LRU-style slot cache.
It is **not** the August hotset/SLRU/SHA timing experiment. Neither experiment
can borrow the other's performance, integrity or memory claims.

## Serving and orchestration

| Published feature / implication | Current evidence and exact limitation |
|---|---|
| One or several LLMs | Single-model operation is valid. Multiple models are optional. The server owns loaded aliases, chat and embedding backends; agentic orchestration belongs to clients |
| OpenAI compatibility | Text chat and embeddings use a subset of the format. Native tools/tool-call parsing, structured outputs, multimodal message arrays and request seed are absent; new usage fields have bounded offline evidence, not full SDK parity |
| Streaming | Shared incremental stop/UTF-8 path implemented in streamed and buffered generation. Offline and real-route/fake-model tests pass; new real-model qualification pending |
| Context / usage | Long prompts truncate by default; `truncate_prompt: false` rejects overflow. Candidate counts retained tokenized prompt, sampled tokens (including terminal EOG/stop) and reused prefix; real-tokenizer gate pending |
| Groups | Parallel, sequential and longest-successful-response fan-out implemented. This is not quality-based voting, continuous batching or proven throughput scaling |
| `retry_once` / `replace_with` | **Incomplete.** One failure leads to a partial outcome or failure; there is no second call or replacement invocation |
| Group KV overrides | **Incomplete.** Parser does not read them; nonempty struct defaults can mask global cache/context settings |
| Pinned / multi-group isolation | **Not enforced as a memory guarantee.** Boot loading and response-quorum decisions exist; `multi-group` aliases pinned-group |
| Generic FIFO, on-demand loading, LRU eviction | **Not established.** Model mutexes serialize inference, not FIFO admission. Discovery is boot-time; no integrated dynamic eviction path |
| Latency limits | Group latency target is not a timeout; health latency remains zero |
| Health / metrics | Process response/uptime and loaded-registry count, not inference readiness. Candidate metric maps are synchronized. Deep health and config reload remain stubs |
| Concurrency | Per-model inference mutex exists. Synchronized metric maps pass concurrent writes/reads and 60 fake-backend HTTP requests pass; real chat/embedding load and cross-endpoint state remain unqualified |

Implementation sources:
[API](../server/api.cpp), [entry point](../server/main.cpp),
[scheduling](../core/scheduling.h), [configuration](../core/config.h),
[model manager](../core/model_manager.h), [backend](../backends/cpu_backend.cpp).
The executable [diagnostic probe](../tests/claims/probe.cpp) characterizes
remaining failures; a passing probe does **not** mean those features work.

## Memory, cache, audit and extension claims

| Claim | Status |
|---|---|
| Adaptive automatic KV / `auto` | **Not operational.** No auto selector, latency telemetry stays zero. Context recreation exists but cache migration, valid reuse state, quality and latency need qualification |
| VRAM reserve, watermarks, group budgets | `reserve_mb` is parsed but unused by loading/request admission; watermarks, isolation and budgets are not parsed. Telemetry alone does not implement resource policy |
| Compression ratio equals whole-system saving | Invalid inference: bit widths describe one storage format. Weights, KV, buffers, alignment and driver accounting must be measured separately |
| Audit hashes / replayable tamper evidence | **Prototype only.** FNV-derived repeated digest, not SHA-256; emitted fields omit reconstruction inputs, only batch path is logged, chain restarts from zero |
| `auth_token` | Parsed, not checked by HTTP handlers. No authentication guarantee |
| Dynamic plugins | Strategy interface exists, dynamic loading is planned |
| Backend portability | Abstraction and build paths exist; not a guarantee of identical kernels, quality or performance across hardware |

No new access-control layer or mandatory integrity machinery was introduced
by this audit. The immediate correction is truthful documentation; implementing
every old design aspiration is neither necessary nor implicitly authorized.

## Performance and research claims

- **Historical desktop:** E2B/E4B speeds are maintainer-reported; no paired
  Ollama baseline. The historical 26B report uses an earlier router plus
  `llama-server`, not proof of current-server speed. Its 8k 99.63 t/s sample
  must not be merged into the 16k 78.68-81.70 t/s range.
- **VRAM:** 26B historical figures are 15,299-15,585 MiB, not that number of
  decimal MB. A reported CPU weight buffer also remains; "fully GPU" means
  offloadable layers, not every model byte. Reported used/free totals are not
  fully reconciled; do not derive a new memory-saving percentage from them.
- **12-task answer-quality summary:** exploratory and not publicly replayable;
  prompts/rubric/raw answers are missing and information/tools differ.
  It does not establish a controlled gain in intrinsic reasoning or zero degradation.
- **August EWS C+:** archived timing result, with resident FFN weights and
  a side transfer arena. Decode timer excludes prefill/loading.
  Simulated per-layer cache policy is not measured physical VRAM residency.
  Seven tracked artifacts match their announced digests, but raw runs and
  ten prompt files named by the manifest are absent from the public bundle.
- **SHA on that experiment:** first read creates the expected digest; later
  reads are compared to it. That is first-observation checking, not verification
  against a pre-trusted signed index. The September runtime has no per-chunk SHA.
- **Cross-model / independent:** another model family is not an independent
  auditor. Mixtral's measured lack of a useful hotset on those traces is not a
  theorem about every workload.
- **Mobile:** numbers are author-reported, without an app bundle, full build/model
  hashes and raw runs. Quantizations differ. NPU/CPU/GPU causal rankings cannot
  be inferred from the table alone.
- **Large MoE / energy:** no GLM 320B inference qualification, no data-center
  GPU-count reduction, no measured power/water savings, and no training-efficiency
  result. These remain research hypotheses.

The README withdraws unpaired "30% less VRAM", "2x faster", universal competitor
rankings, six-model sizing guarantees and unsupported deployment promises.
It does **not** withdraw the September consumed-weight measurements.

## Platform and reproduction boundary

Windows/CUDA/Gemma has the identified local evidence. Linux CUDA operation
and Intel macOS operation are maintainer-reported; ROCm, Apple Silicon and
Android have build paths, not coverage by the Windows campaign.
The standalone Android wrapper is not a complete APK.

The published EWS wrapper requires the supplied patch at the pinned submodule
revision, including for ordinary non-EWS inference builds. Docker recipes do
not yet apply it and may omit runtime libraries. A clean clone-to-inference
packaging test is a remaining deliverable. A no-submodule placeholder build
is never counted as a working inference deployment.
