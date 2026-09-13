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

The 29 vs 60 tok/s spread is unexplained (thermal state, first request after
load, or measurement boundaries); it is reported as observed, not averaged.

## Limits

- One machine, one user, one session; no cold-cache protocol, no dispersion.
- Warm-cache figures depend on the client's deterministic context (KV-reuse);
  a client that rewrites its history each turn will see full prefill instead.
- Not a comparison with llama.cpp or Ollama on the same machine.
- Gemma 4 12B on this hardware is not measured (estimate only: 15–20 tok/s).

## Incident retained

A first start on this machine failed with a TurboQuant KV type (`tq3_1s`):
the engine had run without its preset, taking the built-in default
`type_k/type_v = turbo3`, for which the pinned fork has no Metal kernels.
Fixed in `0e8d824`: on arm64 Apple builds, `turbo*` KV types fall back to
f16 with a logged warning. The default remains `turbo3` on other platforms.
