# Serving functional validation - 8 September 2026

Status: **EIE serving checks pass on the clean Windows/CUDA build. The full
Next client gate fails its offline-attribution scenario; the positive real
12B -> 26B -> 12B exchange and later resident continuation pass.**
Base revision: `97a7ff9a1358544e84599029b7eaa271b8a788f1`.
Tested server source: `d1599a336463b551537a0c49b73fe153d4f7051c`.
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
- The first real HTTP run exposed a nonstream overflow returning HTTP 500.
  Commit `d1599a3` returns HTTP 400 with `context_length_exceeded` instead;
  the existing opt-in overflow rejection is not a new restriction.

No new access controls, resource admission limits or mandatory runtime checks.
Existing strict-model and overflow options remain available.

## Executed checks

Windows 11, MSVC 19.44 / Visual Studio 2022, CUDA 13.2.78, Release x64,
RTX 4090 Laptop (16 GB). Maintainer-side execution, not independent replication.

| Check | Observed result | Boundary |
|---|---|---|
| Offline serving executable | 628 assertions pass | Actual helper/serializer/registry/metrics, not model inference |
| CTest standalone and root `EIE_BUILD_SERVING_TESTS=ON` | 1/1 each passes | Both offline checkout and full initialized clean runtime build tested |
| Real EIE HTTP routes with deterministic fake backend | 12 scenarios pass, plus 60 concurrent requests | SSE/nonstream parity, usage, errors/recovery, catalog/health/metrics; predetermined prompt count |
| Existing claims diagnostic probe | 1/1 passes | Characterizes remaining limitations, not completed features |
| Clean runtime build | Pass | Fresh dependency checkout, published patch; server, EWS forward and native serving-contract targets built with their own DLLs |
| Native 12B gate, normal loading | Pass | Actual tokenizer, streamed/buffered output, stops, prefix/one-shot, callback cancellation, overflow and recovery; F16, 512-token context |
| Native Gemma 26B gate, 16 EWS slots/layer | Pass | Same checks and context; real consumed expert streaming |
| Real HTTP chat + Nomic embeddings | Pass after overflow fix; public runner also re-executed successfully | 12B, F16/512, SSE/stop/usage, 2 x 768 finite nonzero embeddings, idle loaded count 2, 404/400 and recovery |
| Next copied-state client gate | **Fail overall** | Positive auxiliary contribution and resident continuation pass; offline tool invocation/attribution fails, detailed below |
| Whitespace diff check | Pass | Source/document patch |

Commands and fixture entry points are in [tests/serving](../../tests/serving/README.md).
The HTTP runner starts and stops only its own temporary-port fixture.

A separate clean source clone at `97de76f` was created, then updated to `d1599a3`.
Initial dependency setup failed twice: Git's Windows shell could not spawn a
child (`0xC0000142`, `Resource temporarily unavailable`). The failures remain
part of the record; their cause was not established as RAM pressure.

Native `git init`, `fetch --depth 1` and detached checkout obtained the exact
gitlink `2168b0cd8b87c75c29a1e6588692ebbb805b9bd2` from the pinned fork.
Windows long-path handling was required for six upstream UI paths. Applying
`patches/ews-runtime-2168b0.patch` left exactly its expected eight-file diff
(44 additions, 9 deletions), checked with `git -c core.longpaths=true`.
No pre-existing EIE runtime libraries were borrowed for these clean gates.

```powershell
cmake -S . -B build-ews -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_MTMD=OFF -DLLAMA_OPENSSL=OFF -DEIE_BUILD_EWS_TESTS=ON -DEIE_BUILD_SERVING_TESTS=ON
cmake --build build-ews --config Release --target eie-server ews-forward serving-model-contract eie-serving-tests -j 4
ctest --test-dir build-ews -C Release --output-on-failure
```

The initial linked-library preflight is superseded by the clean native tests.
The first real HTTP failure (`http-clean`) is retained alongside the corrected
pass (`http-clean-v2`) and the pass using the public runner (`http-public-runner`).

## Real Next: positive route and a retained failure

Only after the owner stopped production Next, the test copied its memory,
proofs and user-files into a private test directory. It used the actual Next
12B resident (16,384 configured context tokens, its existing model server),
Nomic and vision projector, plus this clean EIE 26B (4,096 context, F16,
16 EWS slots/layer). Vision was initialized, not exercised by an image test.
The test runner was adapted to Next's current Job-owned process API; Next
source, system prompts and production state were not modified.

1. The resident exposed 24 tools and actually called `request_deep_analysis`.
   The 26B returned 172 completion tokens about correlated retry failures.
   The same resident integrated the response and completed its turn.
2. A separate oversized tool request returned `deep-context-too-long`; an
   unknown strict model returned HTTP 404.
3. The test stopped only its owned auxiliary, leaving 12B and Nomic alive.
4. The next request asked for an auxiliary analysis with a local fallback on
   error. The resident produced **no tool call**, but introduced its answer as
   the 26B's analysis. The tool remained exposed; no offline-tool result was
   observed. This is a **false source attribution**, not a successful fallback.
5. A final resident turn answered `17 + 25` with `42`. All turns completed,
   but the unchanged gate `offline_tool_observed` was false, so the runner
   exited with failure. No prompt was retuned or criterion relaxed to make it pass.

The original state digest is unchanged across the run (1,712 files). All owned
test processes were stopped. Raw copied state, conversations and logs remain
private; only the scoped summary and artifact hashes are published here.

**Release boundary:** publish the locally validated EIE serving fixes, while
leaving full Next tool-attribution/recovery qualification explicitly open.
Do not claim the whole system is flawless because its native backend passes.

## Identifiers and measurement limits

Full hashes and bounded results are in the [machine-readable receipt](data/serving-functional-20260908.json).
The final server SHA-256 is
`d0cfc0a01da5171b534b777ccd220ac0085bc936086fc7f9fd1f6de5a1350248`.
The patched runtime is pinned above; build flags and model hashes identify the
tested profile, not a guarantee of byte-reproducible builds on every workstation.

The native tests took 6.422 s (12B) and 30.843 s (26B) for each complete test
process. Those are **not decode speeds**. The 26B native gate counted
61,351,437,312 payload bytes across its sequence of requests; the Next gate
counted 85,598,954,496. These are runtime counters, not PCIe bus measurements,
and must not be relabeled as per-request or decode-only traffic.
No new numerical equivalence, broad quality, energy or platform claim is made.
Earlier EWS measurements keep their original revisions and limits.

## Following milestone

At Alexandre's request, attempt GLM 5.3 Flash as the auxiliary on a fresh Next
copy once this lot is complete. First success means a usable GLM completion and
resident continuation, **without a latency acceptance threshold**. Compatibility,
memory/I/O and full round-trip timings must still be recorded. The GLM port and
model download have not been performed by this serving change.
