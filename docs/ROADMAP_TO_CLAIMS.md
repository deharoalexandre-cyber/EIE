# From current implementation to defensible claims

Status: 7 September 2026. This is an acceptance backlog, not a promise that
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

| Deliverable | Why | Acceptance test |
|---|---|---|
| Clean distribution build | EWS patch and runtime libraries are mandatory dependencies | From a clean clone, apply pinned patch, build normal + EWS targets, load a known model, send chat/embedding requests; archive versions and hashes. Package Docker only after the same test passes inside it |
| Complete streaming contract | SSE plus stop sequences currently suppresses token callbacks | Same prompt streamed/unstreamed, stops spanning token pieces, normal end/length/error/disconnect; same visible result and intelligible recovery |
| Correct context/cache lifecycle | Overflow options exist; adaptive context recreation and prefix reuse need lifecycle coverage | Fresh, repeated-prefix, changed-prefix, one-shot, overflow, interrupted prefill/decode and next-request recovery; compare against fresh-context references |
| Honest counters and readiness | Usage prompt count and health loaded-model count are wrong/incomplete | Prompt tokenizer agrees with usage; loaded registry agrees with health; optional readiness probe exercises inference; metrics withstand concurrent reads/writes |
| Multi-client regression | A model mutex is not complete API concurrency protection | Concurrent chat, embeddings, health and stats; no races, stale aliases, deadlocks or lost errors. Keep normal authorized requests working |

No GPU run was made for this audit; execute the inference parts in a dedicated
test window rather than disturbing a live Next session.

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

GLM 320B remains a separate feasibility campaign, not an advertised supported
model. Before implementation promises:

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
