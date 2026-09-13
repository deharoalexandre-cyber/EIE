# macOS Apple Silicon: first operation receipt, 13 September 2026

**Label: maintainer-reported.** Figures were measured by a Claude Code session
on the tester's machine and relayed by the maintainer; the raw log excerpt
(`ggml_metal_init`, `offloaded … layers`, `[KV]` lines) is not yet retained
here. On 14 September the maintainer confirmed 16 GB RAM and Gemma E2B QAT,
and supplied photographs of the engine/application timing summaries. The
figures below were checked against those visible summaries, not rerun here.
The label remains maintainer-reported until the raw evidence and its exact
execution profile can be inspected. Photos and private conversation content
are not published; only technical measurements are transcribed.

## Setup

| Item | Value |
|---|---|
| Machine | MacBook Pro (2021, as reported), Apple Silicon M1 family, 16 GB unified memory |
| OS | macOS Tahoe 26.6.2 |
| Engine | `eie-server` arm64, static, Metal embedded: EIE `0e8d824`, submodule `2168b0c` + `patches/ews-runtime-2168b0.patch` |
| Engine SHA-256 | `037155fbeedefe8b911d625765fcf5c15ec7bb2336df2da48e4e0ded160c122f` |
| Build | `scripts/build-macos-arm64.sh` (cross-compiled from an Intel Mac; `-DLLAMA_OPENSSL=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_METAL_EMBED_LIBRARY=ON`) |
| Preset | `presets/macos-silicon.yaml`: f16 KV, n_ctx 4096, flash attention off, all layers offloaded |
| Models | `gemma-4-E2B-it-QAT-Q4_0.gguf` (generation), `bge-m3-Q8_0.gguf` (embeddings) |
| Client | Elyne macOS 0.7 (KV-reuse context, SSE streaming) |

The first report called the chip M1 Pro; the maintainer identifies the machine
as a MacBook Pro M1. The exact SoC identifier is not present in the supplied
timing photographs, so the summary uses Apple Silicon/M1 family. The OS,
binary hash, model filenames and build details above remain those of the
original report, not values remeasured during the photograph review.

## Reported figures (single user, interactive session, warm cache)

| Measure | Reported |
|---|---|
| First token, warm KV prefix | 0.02 s |
| Short complete answer | 0.1–0.4 s |
| One paragraph (~150 tokens) | 5.1–5.3 s, ≈ 29 tok/s |
| 400-token answer | 6.6 s, ≈ 60 tok/s |
| Image: Vision OCR + prefill of extracted text + answer | ≈ 3 s |

The "400-token answer" was cut by the client's 400-token cap (finish reason
`length`), so 6.6 s is a capped time, not a completed answer. The 29 tok/s
figure was a completed ~150-token paragraph measured end-to-end, including
memory recall; it is not a decode-rate measurement.

### Engine alone, ~1,636-token history already in the KV cache (same session, later)

| Requested | Generated | Reported wall / adjusted time | First output | Reported output rate |
|---|---|---|---|---|
| 512 | 512 | ≈ 10.5 s (13.0 s measured, 2.6 s queued behind another request) | 0.1 s | 49 tok/s |
| 1024 | 1024 | 21.7 s | 0.07 s | 47 tok/s |
| 2048 | 2048 | 46.4 s | 0.09 s | 44 tok/s |

The 512-token adjusted figure is reproduced as reported, with its rounded
wall/queue components; it is not an independent pure-decode timer. The other
rates are also the displayed summary rates, not newly measured kernel timings.
The earlier short-context sample reports approximately 61 tok/s versus 44 tok/s
for the longer run. These different prompts/cache states do not isolate a
causal effect of answer length. Generating the requested token count alone
does not establish a natural end-of-answer; finish reasons are not supplied
for these three engine-only trials.

### Application end-to-end (real conversation, after the cap was raised)

| Case | Reply | Total |
|---|---|---|
| First message after an engine restart | 788 tokens | 37.3 s total, of which 20.3 s reported generation; about 17 s additional preparation, attributed to history reconstruction by the original report, not an isolated prefill measurement |
| Next long reply | 612 tokens | 22.3 s (14.3 s generation) |
| Short replies | 53–323 chars | 1.2–4.1 s |
| Longer reply without web search | 832 chars | 7.8 s |
| With 1–2 web searches | 224–361 chars | 6.2–10.8 s (1.3–4.0 s per search) |
| With one web search | 775 chars | 11.2 s, including 1.3 s reported search time |

Web search: DuckDuckGo answered 1.2 s when it returned results and 0.3–0.4 s
with an anti-bot page (zero results): two of three test queries from the same
address were blocked. Requests are served one at a time: the tester's own
messages queued behind benchmark requests (16.7 s, 41.6 s, 25.3 s). Benchmarks
must run on a separate engine instance, not on the one in use.

## Limits

- One machine, one user, one session; no cold-cache protocol, no dispersion.
- Warm-cache figures depend on the client's deterministic context (KV-reuse);
  a client that rewrites its history each turn will see full prefill instead.
- Not a comparison with llama.cpp or Ollama on the same machine.
- Gemma 4 12B on this hardware is not measured (estimate only: 15–20 tok/s).
- The approximately 17 s first-request preparation may include history prefill and first-use pipeline creation, but their individual costs were not isolated. It is not a measured prefill throughput; a cold-prefix measurement remains to be done.
- This receipt establishes no EWS-on-Metal or GLM-on-Metal result. The displayed application conversation is not a benchmark of cognition or answer quality.
- With 2,048-token answers, 1.6k of history plus the reply fills ~3,700 of the 4,096-token context; a larger `n_ctx` is needed for that usage.

## Incident retained

A first start on this machine failed with a TurboQuant KV type (`tq3_1s`). Root
cause found on 14 September while producing the Intel receipt: the preset's
`type_k/type_v: f16` was **not applied to the generation model**. Models listed
in a `groups:` entry were loaded with the group's KV override, whose
default-constructed value is `turbo3` (the parser never reads group overrides -
the "Group KV overrides" finding of the claims audit). On Intel the CPU kernels
hide it; on Metal, which has no TurboQuant KV kernels in the pinned fork, the
context fails. Two fixes:

- `0e8d824`: on arm64 Apple builds, `turbo*` KV types fall back to f16 with a warning;
- the bundle revision: group overrides are empty unless set, so the preset's KV type applies
  to every model, on every platform (the `[CPU] loaded: … kv=` log line shows the effective type).

Also corrected in the same revision: the default thread count was a fixed 2,
not half the cores; presets can still force it with `threads:`.

## Upgrade path to *locally verified*

Run `scripts/receipt-macos.sh` on this machine against the published bundle
(its own instance on a test port, never the one in use) and add the JSON under
`data/`; the README label changes in the same commit.
