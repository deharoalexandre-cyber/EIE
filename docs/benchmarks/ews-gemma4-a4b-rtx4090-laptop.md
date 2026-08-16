# EWS field report — Gemma 4 26B-A4B on RTX 4090 Laptop (16 GB)

**Expert-Aware Weight Streaming, de-risking → real engine → pre-registered
holdout verdict. One machine, one day (2026-08-16). Verdict: PASS, bounded.**

Hardware: RTX 4090 Laptop 16 GB (driver 595.79), 32 GB RAM, NVMe Gen4.
Runtime: llama.cpp `fae3a28` (upstream), MSVC 14.44, CUDA. All artifacts,
thresholds and holdouts SHA-256-hashed *before* the data they govern
(manifests in [`data/ews/HASHES.txt`](data/ews/HASHES.txt)).

The initial investigation was triggered by
[AirLLM](https://github.com/lyogavin/airllm)'s weight-streaming approach; the
resulting EWS architecture was developed independently (clean-room C++17, no
code reused) and diverged substantially during measurement. See
[Prior art](../ews/README.md#prior-art--inspirations).

## Method in one line

Every threshold was written and hashed before the first measured token; two
verdicts killed our own designs before one candidate survived on unseen data.

## Campaign timeline (failures included — they carry the information)

| Step | Outcome |
|---|---|
| P0.0 slot-arena | GO — expert slabs substituted into `ggml_mul_mat_id` arenas, bit-identical CPU+CUDA, no ggml fork, no custom kernels |
| P0.3 routing traces, Mixtral 8x7B | **S1 KILL (pre-registered)** — routing entropy 2.997–2.999/3.0 bits: no hotset exists; even Bélády ≈ 1.1 tok/s I/O bound at F=25 % |
| Diagnostic, Gemma-4-A4B (128 exp/top-8) | hotset: 16/128 experts capture 70 % of routing; 32/128 → 86 % |
| P0.4 replication, Qwen3-30B-A3B | same fine-MoE regime, independent family, no shared expert — thesis holds |
| v3 verdict, first cache policy (holdouts G1–G4) | **S2 STOP & REDESIGN (pre-registered)** — policy captured < 50 % of dynamic potential |
| P0.5 policy dev (burned traces only) | practical capture ceiling ≈ 0.40–0.48; predictors fail; SLRU pin-75 % wins |
| P1a bench → real inference | full physical chain measured; router graph-break ≈ 10 µs/layer (negligible); routing-dependency stall named and quantified; speculative prefetch fails (precision 0.095 on *misses*) |
| SHA per chunk | crypto off the miss critical path (plateau at 4/6/8 workers) |
| Engine freeze | 12.6 tok/s dev reference, accounting closed to < 3 ms residual |
| Incident P1-C+-CTX-01 | two original holdouts INVALID (prompt > frozen n_ctx 4096); amendment hashed before replacement contents (mechanical truncation of the *same* material) |
| **C+ verdict, holdouts G5'/G6/G7/G8'** | **PASS 4/4** |

## Final verdict (pre-registered gate C+, EVAL halves only, worst of two runs)

| Holdout | Class | tok/s (worst) | C_engine | C_static | Dynamic gain |
|---|---|---|---|---|---|
| G5' | C++ code analysis | 7.19 | 159.7 MB/tok | 266.9 | −40 % |
| G6 | French prose | 11.10 | 97.1 | 181.8 | −47 % |
| G7 | 25-question battery | 6.55 | 204.1 | 396.6 | −48 % |
| G8' | max supported context | 7.60 | 139.0 | 245.3 | −43 % |

- **C1** (≥ 2 tok/s end-to-end on each): PASS.
- **C2** (engine cold bytes ≤ static hotset, physical bytes, each): PASS.
- **C3** (strict dynamic improvement on ≥ 2): PASS on all four.
- Validity checks: run-pair determinism exact; engine-vs-simulator miss counts
  identical **to the unit** (12 191 / 7 413 / 15 583 / 5 305); accounting
  closure |ΔT_other| ≤ 0.7 ms.

Cold-byte accounting uses physical bytes (frozen 512 KiB chunk layout over
real GGUF offsets) on both sides of the comparison.

## Selected physical measurements

- NVMe direct I/O: 6.6–6.7 GB/s at QD≥4 on 3.35 MB slabs; 3.3 GB/s on 114 MB
  slabs (fine-grained MoE wins on disk too).
- SHA-256 (CNG/SHA-NI): 2.47 GB/s per thread, linear scaling.
- Host→VRAM: ~11 GB/s via `ggml_backend_tensor_set`.
- Router top-k host visibility: 0.29 ms/token over 30 layers.
- Routing-dependency stall (the real bottleneck): ~46 ms/token exposed by
  causal ordering; per-*event* pipeline drain ≈ 0.6 ms is paid per blocked
  layer, not per stalled millisecond.
- A one-byte corruption of a q4_K expert slab can be *behaviorally silent*
  (absorbed by activation quantization): weight integrity cannot be delegated
  to behavioral detection — hence per-chunk SHA-256, fail-closed.

## Scope — read before quoting

Validated: this frozen candidate, on Gemma-4-A4B Q4_0, n_ctx ≤ 4096, on the
hardware above. **Not validated: K3-class (896 experts), any other MoE family,
contexts > 4096, the final signed-index format.** Mixtral-class coarse MoE is
a published negative control, not a supported target. Eligibility of any
future model is decided by routing-profile measurement (per-layer normalized
entropy + hotset curve), not by family or parameter count.
