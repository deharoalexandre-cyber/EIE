# Windows x64 CPU: bundle validation receipt, 2026-09-14

**Label: locally verified.** Produced by `scripts/receipt-windows.py` against the
published bundle `eie-windows-x64-f361b81.zip`, on its own engine instance (port 8099,
not the instance the Elyne Acta application uses on this machine). JSON:
[`data/windows-cpu-20260914.json`](data/windows-cpu-20260914.json).
Maintainer-side execution; not independent replication.

This is the **consumer CPU** line of the repository: a 2020 laptop processor, no GPU used.
It complements, and does not replace, the server-class CPU line expected from a
separate machine, and the Windows/CUDA campaign documented elsewhere.

## Setup

| Item | Value |
|---|---|
| Machine | Laptop, Intel(R) Core(TM) i7-10850H CPU @ 2.70GHz (6 cores / 12 threads), 34.0 GB |
| OS | Windows 11 Pro, build 10.0.22631 (x64) |
| EIE source | `f361b81`; submodule `2168b0c` + `patches/ews-runtime-2168b0.patch` |
| Toolchain | g++ 16.1.0 (MinGW-w64, UCRT, WinLibs), Ninja, `GGML_NATIVE=OFF`, `GGML_OPENMP=OFF`, static link |
| Bundle binary SHA-256 | `20d0ac8e0582ad8493f7af377a0fe8f2db7ec2839873565366b5cb3312fd390c` |
| Preset | `windows-cpu.yaml`: n_ctx 4096, KV f16/f16, flash_attn false, 0 layers offloaded (CPU) |
| Effective per model (from the engine log) | Qwen3.5-9B-Q6_K: kv=f16/f16 ctx=4096 threads=6, nomic-embed-text-v2-moe.Q6_K: kv=f16/f16 ctx=4096 threads=6 |
| Models | Qwen3.5-9B-Q6_K `3000f1b36c1c4cf6…` (9.20 B parameters, 7.6 GB), nomic-embed-text-v2-moe.Q6_K `e74dafe6932fc7ae…` |
| Load until both models healthy | 20.25 s (warm page cache) |

The models are the ones already installed on this machine for Elyne Acta, not the Gemma 4
E2B QAT Q4_0 / bge-m3 pair used by the macOS and Android receipts. A 9B Q6_K model on six
laptop cores is a heavier profile than those receipts; the figures are not comparable
across platforms.

## Measurements

Embeddings: 768 dimensions, L2 norm 1.0, 0.100 s wall.

| Request | prompt tokens | cached prefix | generated | engine time | queue + transport | tok/s (engine) |
|---|---|---|---|---|---|---|
| cold prefix, 64 max | 174 | 0 | 64 (length) | 37.12 s | 7 ms | 1.7 |
| warm prefix, 64 max | 264 | 0 | 64 (length) | 44.97 s | 8 ms | 1.4 |
| warm prefix, 256 max | 351 | 0 | 256 (length) | 110.73 s | 18 ms | 2.3 |

`engine time` is the server-side request time from the `[KV]` log line (prefill plus
generation); `tok/s (engine)` divides generated tokens by that time. Because no prefix was
reused (next section), every row includes a full prefill, so the 256 row is only a lower
bound on the decode rate. Queue/transport is wall time minus engine time on localhost.

## Prefix reuse did not apply to this model

All three requests report `reused=0` although the second and third turns extend the
first conversation. Qwen3.5 is a hybrid architecture (recurrent Gated DeltaNet layers
alongside attention): the backend's partial cache removal (`llama_memory_seq_rm` from the
matched prefix onwards) is refused by the runtime for recurrent state, and the backend
falls back to re-evaluating the whole prompt. This is the documented fallback path in
`backends/cpu_backend.cpp`, not a failure of the bundle; on models with a plain KV cache
(Gemma 4 E2B in the macOS receipts) the warm rows show the reused prefix. A Windows run
with the reference Gemma/bge-m3 pair is the natural follow-up to measure the reuse on this
machine.

## Limits

- one machine, one session, warm page cache for the model files
- engine_ms is the server-side request time from the [KV] log line; queue_and_transport_ms is wall minus engine
- tokens_per_engine_second includes prefill of the new turn; with no prefix reuse it understates the decode rate
- no comparison with llama.cpp or Ollama on the same machine
- no cold-cache protocol, no dispersion
- consumer laptop under Windows power management; no thermal or power control
