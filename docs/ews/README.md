# EWS — Expert-Aware Weight Streaming (experimental)

**Status: validated v1 candidate (frozen, archived). Not yet integrated into
the EIE server — this documents the design and the evidence behind it.**

EWS lets EIE run Mixture-of-Experts models whose weights exceed available
VRAM by streaming expert weights from NVMe at *expert granularity*, instead
of layer granularity (the classic AirLLM approach — concept credited, no code
reused; clean-room C++17).

## Central finding

> **EWS eligibility is governed by the cold working set induced by routing —
> not merely by total model size or by being MoE.**

Coarse, load-balanced MoE (Mixtral 8x7B: 8 experts, near-maximal routing
entropy, 114 MB per expert slab) has *no exploitable locality*: even a
clairvoyant cache stays below the viability floor. Fine-grained MoE
(Gemma 4 26B-A4B: 128 experts / top-8 / 3.3 MB slabs; independently
replicated on Qwen3-30B-A3B) concentrates 70–86 % of routed traffic on a
small per-layer hotset — the storage footprint decouples from the
instantaneous working set.

A second empirical result from the final holdout verdict:

> **The static hotset provides structure; the dynamic cache provides
> robustness outside the calibration distribution.**
> On unseen workloads with short calibration, the pure static hotset degrades
> to 0.51–0.77 hit rate while the frozen hotset+SLRU policy recovers
> 0.75–0.88, cutting cold bytes per token by 40–48 %.

## Frozen v1 candidate

| Component | Frozen value |
|---|---|
| Cache policy | 75 % pinned hotset (calibrated) + 25 % SLRU |
| VRAM budget | F = 25 % of expert store (per layer) |
| Fetch | 512 KiB aligned chunks, task granularity = chunk |
| Integrity | independent SHA-256 per chunk, verified against a digest table; slot READY only after *all* chunks validate (fail-closed) |
| Router visibility | graph-break read of top-k, ≈10 µs/layer (measured negligible) |
| Known limitation | n_ctx ≤ 4096 (compiled into the frozen candidate) |

Rejected by measurement during the campaign: layer-granularity streaming,
routing predictors (three independent failures: byte savings, EWMA next-use,
speculative bigram prefetch — a good *routing* predictor is a poor *miss*
predictor), TinyLFU admission, sliding hotsets (prohibitive churn), deep I/O
queues as a latency cure (the routing-dependency stall is latency-bound, not
bandwidth-bound).

## Headline result (pre-registered, end-to-end, unseen holdouts)

> **Gemma 4 26B-A4B Q4_0 — EWS frozen candidate**
> 4/4 unseen holdouts passed (pre-registered C+ gate)
> **6.55–11.10 tok/s end-to-end** (worst-of-two-runs, EVAL half only)
> **40–48 % fewer cold bytes/token vs static hotset**
> SHA-256 chunk verification fail-closed enabled
> Scope: Gemma-4-A4B, n_ctx ≤ 4096, tested hardware only

**No claim is made for K3-class models, other MoE architectures, or contexts
above 4096 tokens.** Mixtral-class coarse MoE is explicitly out of scope
(published negative control).

## Prior art / inspirations

EWS was initially inspired by [AirLLM](https://github.com/lyogavin/airllm)'s
work on weight streaming (layer-wise streaming in 2023–2024, then per-expert
streaming in its 2026 revival). **AirLLM is credited for the general streaming
approach.** EWS is an independent clean-room C++17 implementation with a
different execution model (slot-arena substitution inside `ggml_mul_mat_id`),
cache policy (calibrated hotset + SLRU), integrity layer (fail-closed
per-chunk SHA-256 against a signed digest table) and empirical eligibility
criteria (routing-profile measurement). **No AirLLM code has been reused.**

Related work also includes expert-offloading approaches such as
PowerInfer (SJTU) and the Mixtral offloading research by Eliseev & Mazur
(LRU expert cache + speculative prediction). llama.cpp's mmap/layer offload
serves as the comparison baseline.

## Reports and artifacts

Full campaign report, including the failures that shaped the design:
[`docs/benchmarks/ews-gemma4-a4b-rtx4090-laptop.md`](../benchmarks/ews-gemma4-a4b-rtx4090-laptop.md).
Raw protocols, manifests, verdict and reproduction scripts:
[`docs/benchmarks/data/ews/`](../benchmarks/data/ews/).
