# From current implementation to defensible claims

Status: updated 8 September 2026. This is an acceptance backlog, not a promise that
every possible feature will be built. Removing an unnecessary claim is a
valid resolution. Priority is useful working inference, not more blocking
layers.

## Already locally achieved and now published

- Consumed expert-weight streaming, actual byte/cache counters and physical
  weight-buffer savings on Gemma 26B.
- Paired numerical gates, including an intentionally corrupted-weight pilot
  control; short-profile bit-identical logits.
- Coexistence with real Next 12B on copied state, with tool calls in the
  second run and the first unsuccessful tool-use run retained.
- Target-runtime integration, corrected KV-name mapping, real device-memory
  queries, explicit auxiliary request errors and prefill-disconnect fix.

These are not future milestones. [Evidence and exact scope](benchmarks/ews-consumed-20260905.md).

## First: eliminate practical serving surprises

The [8 September serving lot](benchmarks/serving-functional-20260908.md) implements
incremental stop/UTF-8 output, usage counters, loaded-model counts and synchronized
metrics. It also fixes stopped-token KV bookkeeping and exception-result
construction. Offline tests, a clean runtime build, real 12B/26B native gates and
real HTTP chat/embedding checks pass. The actual Next auxiliary contribution
and resident continuation also succeed, but **the full Next gate fails**: in
the offline-followup scenario, the resident attributes an answer to the stopped
26B without calling it. Preserve that failure; repair/test source attribution
at the client separately from EIE's successful serving checks.

| Deliverable | Why | Acceptance test |
|---|---|---|
| Clean distribution build | Windows/CUDA source build and real inference passed; packaging is still platform-specific | Retain the recorded build/model hashes and dependency workaround; package Docker only after the same chat/embedding test passes inside it |
| Complete streaming contract | Real 12B/26B output/recovery gates passed on the recorded profile | Broaden prompts and clients: streamed/unstreamed, stops spanning token pieces, normal end/length/error/disconnect; same visible result and intelligible recovery |
| Correct context/cache lifecycle | Overflow options exist; adaptive context recreation and prefix reuse need lifecycle coverage | Fresh, repeated-prefix, changed-prefix, one-shot, overflow, interrupted prefill/decode and next-request recovery; compare against fresh-context references |
| Honest counters and readiness | Real-tokenizer and loaded-registry checks passed; inference readiness remains distinct | Expand model/load coverage; any future readiness probe must exercise inference rather than rename process health |
| Multi-client regression | A model mutex is not complete API concurrency protection | Concurrent chat, embeddings, health and stats; no races, stale aliases, deadlocks or lost errors. Keep normal authorized requests working |

The 8 September GPU checks used a dedicated window after the owner stopped
production Next. Test processes were stopped and the original state digest was
unchanged. They do not qualify all of Next's cognition or source attribution.

## Then: make advertised group behavior real, or remove the option

| Deliverable | Acceptance |
|---|---|
| Real retry/replacement | Inject first-call failure: retry invokes exactly one second call; replacement invokes the declared alternate model; original error and final result stay observable |
| Explicit group configuration inheritance | Global settings apply when no override exists; per-group settings parse and override only specified fields; verify actual instantiated context/cache |
| Group policies | Separate strict/partial/quorum, longest-response fan-out and sequential chaining tests; multi-group should have distinct behavior before a distinct guarantee is advertised |
| Memory policy, only if retained as a feature | Parse settings, feed actual device measurements into the real loader, exercise representative fits/pressure/recovery; document fallback and uncertainty. Do not silently turn speculative budgets into blanket request blockers |
| Adaptive KV, only if retained | Measured trigger, supported cache transition, correctly invalidated reuse state, reference-quality checks and bounded recovery latency. An `auto` label requires an implemented selection policy |

Pinned reservation, dynamic loading, eviction and latency enforcement are
separate capabilities; one should not be inferred from another.

## Expand EWS evidence without discarding the achieved result

1. Publish a compact independently replayable numerical bundle: exact source
   revisions, build profile, model hash, prompt hashes, commands and small
   reference outputs. The current summaries plus local raw hashes are valuable,
   but a hash is not a downloadable logit tensor.
2. Repeat timings with a declared warm/cold-cache procedure and paired
   references. Report prefill, first-token, decode and complete request wall
   time separately; add dispersion and peak-memory sampling.
3. Expand the frozen test corpus and contexts, separating prefill from decode.
   Keep negative controls and failed cases, not just successful outputs.
4. Qualify the whole Next envelope: representative long prompts, actual vision
   inference, consolidation/reflection events, belief revision and return from
   auxiliary inference. The September short envelope did not exercise all of
   these functions.
5. Qualify other quantizations and hardware separately. Windows direct I/O
   results do not validate Linux page-cache behavior, ROCm, Metal or mobile.
6. Benchmark policy changes against equal budgets: slot capacity, RAM cache,
   resident model/context and instrumentation must match. Do not combine the
   August simulated SLRU claims with the September consumed LRU implementation.

## GLM and resource-efficiency research

GLM 320B remains a separate experimental runtime path, not a generally qualified
model on EIE's default build. The first **functional, not fast** milestone passed on 8 September using
a separate native runtime: actual fresh-state Next 12B -> GLM tool call -> 12B
continuation, with a complete 188-token auxiliary answer. See the
[native receipt and retained truncated first run](benchmarks/glm53-native-next-20260908.md).
Latency has no speed pass/fail threshold. The 26B baseline and production Next
state remain unchanged. This is a CPU-expert mmap baseline, not a GLM EWS result.

The shared EWS reader now handles split files, separate gate/up/down projections,
metadata-derived routed layers and expert counts, and physical slots with
unchanged logical router IDs. The [GLM-specific runtime patch](GLM_EWS_EXPERIMENT.md)
targets the downloaded/rehashed GLM artifact and a separate pinned native fork;
EIE's default llama.cpp pin still lacks GLM-5-Next. The 42 routed trunk layers /
288 experts / 8-slot host cache pass a short paired numerical gate. Updated-reader
Gemma regressions remain on the original runtime. Neither production Next nor
its runtime is replaced by the experiment.

The [EIE/EWS online Next gate](benchmarks/glm53-ews-next-20260908.md) is also
complete: real tool invocation, normal GLM completion, resident synthesis and
subsequent resident turn. It is a short, fresh-state functional result with
strong RAM pressure, not a clean brevity/quality result or the whole Next
lifecycle qualification. Keep both earlier failed attempts in the history.

The [first hybrid GPU placement](benchmarks/glm53-ews-gpu-next-20260908.md) now
passes: 18 routed expert layers on GPU, 24 on CPU, with actual Next cohabitation.
The native/EWS GPU control covers two routed layers; three larger-placement
comparisons establish cache-size consistency, not full native equivalence.
The complete 148-token live answer follows an earlier truncated attempt, which
remains visible. No paired speedup or isolated token-budget effect is claimed.

Next, measure whether more expert computation can move to GPU while retaining
resident headroom, using explicit placement budgets and new numerical controls.
All 42 expert layers on GPU are not yet qualified. Reduce repeated slab reads
with a measured cache policy before promising usable long-context latency.
Measure whole-process/whole-host memory and I/O, not just expert tensors.

After initial functional bring-up, the fuller measurement campaign can:

- verify the exact model artifact and runtime architecture support;
- specify a hashed calibration/holdout corpus, route instrumentation and its
  overhead, prefill/decode separation, hard expert-slot coverage versus router
  probability mass, and hardware/storage budgets;
- measure hotset/transfer requirements rather than assuming a Pareto law;
- choose feasibility thresholds from a stated device budget and service target;
- compare with actual CPU/offload baselines and count all initialization,
  I/O, RAM, VRAM and latency costs.

Energy or training-hardware claims require their own matched measurements.
Inference I/O savings are not automatically fewer training GPUs, lower water
use or lower data-center electricity.

## Optional, not a condition for a useful EIE

Dynamic plugins, a cryptographic audit ledger, a broader admin plane and a
six-model preset are optional product choices. The current audit logger is a
prototype and token authentication is not implemented; keeping those limits
explicit is preferable to adding unrelated machinery as a release prerequisite.

A milestone is complete when the described behavior succeeds and the evidence
is retained, not when its configuration key exists or its unit test count grows.
