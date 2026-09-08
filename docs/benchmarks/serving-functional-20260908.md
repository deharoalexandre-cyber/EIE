# Serving functional candidate - 8 September 2026

Status: **offline checks passed; new real-model qualification pending**.
Base revision: `97a7ff9a1358544e84599029b7eaa271b8a788f1`.
Work branch: `fix/serving-functional-20260908`.

## Changes

- Shared incremental stop matching and UTF-8 output for streamed/buffered chat.
  Stops can span token pieces or end inside one; unresolved prefixes are withheld
  until decidable, rather than suppressing the entire stream.
- Actual retained prompt token count, sampled completion count and reused-prefix
  detail in both response modes. JSON control characters are escaped.
- Health/metrics loaded count comes from the model registry, including idle and
  embedding models. Metric reads/writes use the same mutex.
- Only successfully decoded generated tokens enter persistent KV bookkeeping.
  One-shot decode failure does not clear another context's reuse metadata.
- Backend exceptions become an explicit failed result; positional aggregate
  initialization no longer attempts to construct a string from a null pointer.

No new access controls, resource admission limits or mandatory runtime checks.
Existing strict-model and overflow options are unchanged.

## Executed checks

Windows 11, MSVC 19.44 / Visual Studio 2022, Release x64.

| Check | Observed result | Boundary |
|---|---|---|
| Offline serving executable | 628 assertions pass | Actual helper/serializer/registry/metrics, not model inference |
| CTest standalone and root `EIE_BUILD_SERVING_TESTS=ON` | 1/1 each passes | Root checkout has no initialized submodule; root server is a placeholder |
| Real EIE HTTP routes with deterministic fake backend | 11 scenarios pass, plus 60 concurrent requests | SSE/nonstream parity, usage, errors/recovery, catalog/health/metrics; predetermined prompt count |
| Existing claims diagnostic probe | 1/1 passes | Characterizes remaining limitations, not completed features |
| Inference-enabled server and model-contract executable | Compile and link pass | Candidate wrapper linked to existing locally patched runtime libraries; not a clean runtime rebuild |
| Whitespace diff check | Pass | Source/document patch |

Commands and fixture entry points are in [tests/serving](../../tests/serving/README.md).
The HTTP runner starts and stops only its own temporary-port fixture.

A separate clean source clone at `97de76f` was created. Dependency initialization
did not complete: Git's Windows shell failed to spawn a child process
(`0xC0000142`, `Resource temporarily unavailable`). This is a failed setup
attempt, not a successful clean build or evidence of a model/runtime defect.

## Still to execute

1. On the known 12B and streamed Gemma 26B, run `serving-model-contract`:
   compare the actual tokenizer and streamed/buffered results, stop handling,
   one-shot/prefix reuse, cancellation and next-request recovery.
2. Build the pinned and patched runtime from a clean clone, rather than borrow
   the existing compiled libraries; validate chat and embeddings.
3. Use only a copy of Next to exercise the actual 12B -> 26B -> 12B route,
   including a subsequent resident turn.

Production Next and its auxiliary model remained active during the offline
checks. Their processes, state and binaries were not replaced or stopped.
No new GPU inference or latency claim is made by this receipt. Earlier EWS
measurements remain valid for their recorded revisions, not proof that this
candidate passed the outstanding gates.

## Following milestone

At Alexandre's request, attempt GLM 5.3 Flash as the auxiliary on a fresh Next
copy once this lot is complete. First success means a usable GLM completion and
resident continuation, **without a latency acceptance threshold**. Compatibility,
memory/I/O and full round-trip timings must still be recorded. The GLM port and
model download have not been performed by this serving change.
