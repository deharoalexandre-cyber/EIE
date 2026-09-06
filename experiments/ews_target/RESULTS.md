# EIE runtime port + real Next envelope — 2026-09-05

**Outcome: the consumed-weight path runs inside EIE, concurrently with the
real Next application composition. This is a bounded integration result,
not a production rollout or a GLM feasibility result.**

Machine: RTX 4090 Laptop 16 GB, driver 595.79, Windows, CUDA 13.2,
MSVC 19.44. EIE base `67a779d3b291bed29a09a43330f552cfbe084277`, llama.cpp
pin `2168b0cd8b87c75c29a1e6588692ebbb805b9bd2` plus the archived patch.
Source, model and binary hashes: [published evidence](../../docs/benchmarks/data/ews-runtime-integration-20260905.json).
These hashes identify the measured September 5 state, not every subsequent
source revision. Publication adds later runtime fixes; no new GPU campaign
is implied. See the [publication receipt](../../docs/benchmarks/claims-verification-20260907.md).

## Numerical gate

Two existing Gemma4 prompts (French and code), 32 greedy predictions each,
full vocabulary logits saved. Full-weight references and streamed candidates
use the same one-token microbatch, F16 cache, router callback boundaries and
disabled CUDA graphs. The final build was tested again after the Windows
declaration compatibility adjustment.

| Candidate | French | Code |
|---|---|---|
| 32 expert slots/layer | Bit-identical | Bit-identical |
| 8 expert slots/layer | Bit-identical | Bit-identical |

33,554,432 FP32 values compared across four candidate/reference comparisons;
maximum absolute difference **0**. Generated token sequences also match.
This does not establish correctness on all inputs or at long contexts.

## Actual Next composition

Two runs of five turns each, using Next's real code and an exact copy of its
memory, proofs and user files. Original and initial-copy digests match across
1,528 files. Original digests are unchanged after both runs.

| Property | Observed |
|---|---|
| Next resident | Actual Gemma4 12B QAT, configured context 16,384 |
| Prompt lengths | Up to 9,494 prompt-evaluation tokens in the logged sessions |
| Embedding | Nomic established before, during and after EWS |
| Vision | Established before, during and after EWS; no image inference tested |
| EIE model | Gemma4 26B-A4B Q4_0, 16 slots/layer, context 2,048, F16 KV |
| Concurrent 26B work | Four HTTP responses in total, 192 generated tokens each |
| 26B request wall time | 32.17–33.34 seconds, including prefill and contention |
| Output / total request time | 5.76–5.97 tokens/s; **not isolated decode throughput** |
| External GPU sampling | Peak 13,408 MiB used; minimum 2,641 MiB free |
| Return after unloading EIE | Next completed a further turn in each run |

No production Next binary, configuration, journal or memory was replaced.
The test-owned processes were stopped at the end. Lifecycle drivers were
enabled with production policies, but this short run did not exercise an idle
consolidation or belief-reflection event. It is not a soak test.

### Tool-use result: retain the failed first attempt

The first run completed its turns and performed hybrid retrieval, but the
model made **zero native tool calls**, despite 23 tools being exposed. It also
invented filenames in its answer. That run is **not** a successful tool-use
test. Runtime stability and answer/tool-use correctness are different gates.

The second run explicitly asked for tool execution. The journal confirms
`completed` outcomes for `inspect_runtime`, `search_memory`, `search_beliefs`
and `list_files`. The three latter tools ran while the 26B workload was active.
Actual file-list output was then used in the answer.

`search_beliefs` returned no matches for the chosen multi-term query, so the
requested `inspect_belief` follow-up did not run. The final requested
`inspect_runtime` re-call also did not run; post-unload liveness is instead
attested by the runtime snapshot and successful final turn. Neither missing
call is counted as passed. No belief was revised by this test. The correctness
of existing memories and the model's factual interpretation was not evaluated.

## HTTP and cache regression checks

On the final EIE build, in **both F16 and Turbo3**:

- Normal chat → one-shot → repeated normal chat produce identical text.
- SSE output matches non-SSE text and terminates correctly.
- KV-prefix reuse occurs (22 tokens in this fixture).
- Actual EWS callbacks and file reads are nonzero; cache accounting balances.
- Physical expert bytes equal 16/128 of logical expert bytes.
- The OpenAI model catalog keeps its original schema.
- Requested KV type is used without an F16 fallback.

The Turbo3 smoke is not a numerical-quality or F16-equivalence claim.
The three KV type-name mappings were corrected without modifying quantizers
or CUDA kernels. The compiled HTTP dependency and MSVC declaration were also
fixed to build the exact pinned runtime.

## What remains outside this result

Automatic cognitive delegation from Next to the 26B is **not added**: Next and
EIE are independent, simultaneously active services. Full 16K saturation,
longer 26B contexts, long-duration background activity, image inference,
other architectures/quantizations, LoRA and non-Windows backends remain
unqualified. This is not the older hotset/SLRU/SHA campaign and its numbers
must not be merged with that campaign's scores.

Build/run instructions: [runtime-port.md](../../docs/ews/runtime-port.md).
Private raw evidence remains under `build-ews/target-forward-final`,
`build-ews/next-envelope-v1`, `build-ews/next-envelope-v2` and
`build-ews/target-api-v1`; it includes conversation-derived content and must
not be published with the shareable summary.
