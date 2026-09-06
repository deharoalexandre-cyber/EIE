# EWS - Expert-Aware Weight Streaming (experimental)

## Current status: consumed runtime published

The September implementation runs expert matmuls on the slabs loaded into
bounded per-layer slots. It is integrated into EIE's target runtime.

- [September consumed-weight measurements](../benchmarks/ews-consumed-20260905.md)
- [Build, patch and runtime guide](runtime-port.md)
- [Target integration and real Next envelope](../../experiments/ews_target/RESULTS.md)
- [Remaining qualification work](../ROADMAP_TO_CLAIMS.md)

Local Gemma 26B numerical evidence and real Next coexistence are established
within the documented profiles. GLM, arbitrary contexts, other architectures
and other hardware are not thereby validated. This runtime uses an LRU-style
slot cache, one-token microbatches and no per-chunk SHA verification.

## Historical August routing / I/O campaign

The [August report](../benchmarks/ews-gemma4-a4b-rtx4090-laptop.md) and
[frozen archive](../benchmarks/data/ews/README.md) are preserved for the
sequence of experiments, including negative results.

**Important distinction:** the frozen C+ timing engine transferred bytes into
a side arena while FFNs still consumed resident weights. Its simulated
per-layer cache policy was not the current physical consumed-slot path.
"End-to-end" in that archive means timed decode including injected I/O,
not loading, prefill or complete application latency.

The useful hypothesis is that routing-induced cold working set, not total
model size alone, governs streaming feasibility. On the reported Mixtral
traces at the tested budget, even an oracle cache missed the viability target.
This is not proof of no locality for every Mixtral workload. Gemma and a
separate Qwen family showed stronger concentration on the examined traces;
another model family is not an independent replication team.

### Archived candidate, not current runtime configuration

| Item | August experiment |
|---|---|
| Policy | 75% pinned calibrated hotset + 25% SLRU |
| Logical cache budget | 25% of each layer's expert store |
| Transfer | 512 KiB aligned chunks into a side arena |
| Integrity | Expected SHA-256 first populated from the first read, subsequent reads compared; no pre-trusted signed digest index |
| Scope | Frozen 4,096-token context, reported Gemma holdouts |
| Reported decode result | 6.55-11.10 tok/s, EVAL half, worst of two |
| Reported dynamic gain | 40-48% fewer cold bytes than the smaller static pinned portion alone |

The static comparison has fewer allocated slots than the combined policy;
it measures the dynamic portion's incremental contribution, not a comparison
to an equal-capacity fully static cache. Raw timings/traces and some manifest
prompts are not included, so full independent score recomputation is not
available from the archive alone.

## Prior art / inspirations

[AirLLM](https://github.com/lyogavin/airllm) is credited for inspiring weight
streaming. The maintainers state that their C++ implementation was developed
independently without reusing AirLLM code; this audit did not establish legal
provenance or novelty. It also does not independently verify historical
competitor capability or performance claims.

Related approaches include PowerInfer, expert offloading/cache work by
Eliseev and Mazur, and llama.cpp's offload mechanisms. Refer to exact versions
and matched measurements when comparing systems.

The August SHA/hotset experiment and September slot substitution are
different implementations. There is no implemented signed-index integrity
guarantee to attribute to the current runtime.
