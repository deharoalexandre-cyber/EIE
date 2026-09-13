# macOS Apple Silicon — first operation receipt, 13 September 2026

**Label: maintainer-reported.** Figures were measured by a Claude Code session
on the tester's machine and relayed by the maintainer; the raw log excerpt
(`ggml_metal_init`, `offloaded … layers`, `[KV]` lines) is not yet retained
here. When it is appended, this receipt becomes *locally verified*.

## Setup

| Item | Value |
|---|---|
| Machine | MacBook Pro M1 Pro (2021), 16 GB unified memory |
| OS | macOS Tahoe 26.6.2 |
| Engine | `eie-server` arm64, static, Metal embedded — EIE `0e8d824`, submodule `2168b0c` + `patches/ews-runtime-2168b0.patch` |
| Engine SHA-256 | `037155fbeedefe8b911d625765fcf5c15ec7bb2336df2da48e4e0ded160c122f` |
| Build | `scripts/build-macos-arm64.sh` (cross-compiled from an Intel Mac; `-DLLAMA_OPENSSL=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_METAL_EMBED_LIBRARY=ON`) |
| Preset | `presets/macos-silicon.yaml` — f16 KV, n_ctx 4096, flash attention off, all layers offloaded |
| Models | `gemma-4-E2B-it-QAT-Q4_0.gguf` (generation), `bge-m3-Q8_0.gguf` (embeddings) |
| Client | Elyne macOS 0.7 (KV-reuse context, SSE streaming) |

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

| Requested | Generated | Wall time | First token | Decode rate |
|---|---|---|---|---|
| 512 | 512 | ≈ 10.5 s (13.0 s measured, 2.6 s queued behind another request) | 0.1 s | 49 tok/s |
| 1024 | 1024 | 21.7 s | 0.07 s | 47 tok/s |
| 2048 | 2048 | 46.4 s | 0.09 s | 44 tok/s |

Decode rate declines gently with answer length (61 → 44 tok/s from a short
context to 2,048 generated tokens on top of 1.6k history).

### Application end-to-end (real conversation, after the cap was raised)

| Case | Reply | Total |
|---|---|---|
| First message after an engine restart | 788 tokens | 37.3 s, of which 20.3 s generation — the ~17 s remainder is the full history prefill (≈ 1.6k tokens), done once, then reused |
| Next long reply | 612 tokens | 22.3 s (14.3 s generation) |
| Short replies | 53–323 chars | 1.2–4.1 s |
| With 1–2 web searches | 224–361 chars | 6.2–10.8 s (1.3–4.0 s per search) |

Web search: DuckDuckGo answered 1.2 s when it returned results and 0.3–0.4 s
with an anti-bot page (zero results) — two of three test queries from the same
address were blocked. Requests are served one at a time: the tester's own
messages queued behind benchmark requests (16.7 s, 41.6 s, 25.3 s). Benchmarks
must run on a separate engine instance, not on the one in use.

## Limits

- One machine, one user, one session; no cold-cache protocol, no dispersion.
- Warm-cache figures depend on the client's deterministic context (KV-reuse);
  a client that rewrites its history each turn will see full prefill instead.
- Not a comparison with llama.cpp or Ollama on the same machine.
- Gemma 4 12B on this hardware is not measured (estimate only: 15–20 tok/s).
- The 17 s first prefill (≈ 96 tok/s) is far below what Metal should reach for this model; it may include first-use pipeline creation. Cold-prefix prefill throughput is a separate measurement still to do.
- With 2,048-token answers, 1.6k of history plus the reply fills ~3,700 of the 4,096-token context; a larger `n_ctx` is needed for that usage.

## Incident retained

A first start on this machine failed with a TurboQuant KV type (`tq3_1s`):
the engine had run without its preset, taking the built-in default
`type_k/type_v = turbo3`, for which the pinned fork has no Metal kernels.
Fixed in `0e8d824`: on arm64 Apple builds, `turbo*` KV types fall back to
f16 with a logged warning. The default remains `turbo3` on other platforms.
