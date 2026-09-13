# macOS Intel — bundle validation receipt, 2026-09-14

**Label: locally verified** — produced by `scripts/receipt-macos.sh` against the
published bundle, on its own engine instance (port 8091). JSON:
[`data/macos-intel-20260914.json`](data/macos-intel-20260914.json).
Maintainer-side execution; not independent replication.

## Setup

| Item | Value |
|---|---|
| Machine | MacBookPro15,2 — Intel(R) Core(TM) i5-8279U CPU @ 2.40GHz, 8.6 GB |
| OS | macOS 15.7.9 (x86_64) |
| EIE source | `03be806`; submodule `2168b0c` + `patches/ews-runtime-2168b0.patch` |
| Bundle binary SHA-256 | `1b092f2782b3ac0ff2d50dea5b3b97d456e35710574abe6d08d9d0c4dfd2f09c` |
| Preset | `macos-cpu.yaml` — n_ctx 4096, KV f16/f16, flash_attn false, backend 0 (CPU) |
| Effective per model (from the engine log) | gemma-4-E2B-it-QAT-Q4_0: kv=f16/f16 ctx=4096 threads=4, bge-m3-Q8_0: kv=f16/f16 ctx=4096 threads=4 |
| Models | gemma-4-E2B-it-QAT-Q4_0 `aa6eb6d481b583a3…`, bge-m3-Q8_0 `950f4a8e5e19477a…` |
| Load until both models healthy | 5.25 s (warm page cache) |

## Measurements

Embeddings: 1024 dimensions, L2 norm 1.0, 0.108 s wall.

| Request | prompt tokens | cached prefix | generated | engine time | queue + transport | tok/s (engine) |
|---|---|---|---|---|---|---|
| cold prefix, 64 max | 172 | 0 | 64 (length) | 8.86 s | 5 ms | 7.2 |
| warm prefix, 64 max | 264 | 236 | 64 (length) | 5.76 s | 8 ms | 11.1 |
| warm prefix, 256 max | 351 | 328 | 245 (stop) | 20.53 s | 13 ms | 11.9 |

`engine time` is the server-side request time from the `[KV]` log line (prefill of
the non-cached tokens plus generation); `tok/s (engine)` divides generated tokens by
that time, so the warm 256 row is the closest to a decode rate. Queue/transport is
wall time minus engine time on localhost.

## What changed versus earlier operation on this machine

Before this revision the same machine ran the same model with `kv=turbo3/turbo3` and
2 threads despite the preset (group override masking + fixed thread default); typical
warm generation was 5–6 tok/s. With f16 KV and 4 threads it is ~12 tok/s. This is a
before/after on one machine, not a controlled comparison.

## Limits

- one machine, one session, warm page cache for the model files
- engine_ms is the server-side request time from the [KV] log line; queue_and_transport_ms is wall minus engine
- tokens_per_engine_second includes prefill of the new turn; warm_prefix_256 approximates decode rate
- no comparison with llama.cpp or Ollama on the same machine
- no cold-cache protocol, no dispersion
