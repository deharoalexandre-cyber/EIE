# Gemma 4 26B A4B on a 16 GB RTX 4090 Laptop

This is a single-system field report for running Google's
[Gemma 4 26B A4B instruction model](https://huggingface.co/google/gemma-4-26B-A4B-it)
through EIE's GGUF router on a 16 GB laptop GPU. It is intended as a
reproducible deployment reference, not as a standardized model-quality
benchmark.

## Result

The QAT Q4_0 model fits fully on the GPU with a 16,384-token context when
both KV caches use Q4_0. In this configuration, observed generation speed
was 79-100 tokens/s and long-prompt evaluation was 2,442-2,614 tokens/s.

| Metric | Observed result |
| --- | ---: |
| GPU memory used | 15,299-15,585 MiB |
| GPU memory free | 464-750 MiB |
| CUDA model buffer | 13,755.42 MiB |
| CPU-mapped model buffer | 577.50 MiB |
| Non-SWA KV cache, Q4_0 | 90.00 MiB |
| SWA KV cache, Q4_0 | 84.38 MiB |
| Prompt evaluation, ~4k-token prompts | 2,442-2,614 tokens/s |
| Generation, representative 16k-context requests | 78.7-81.7 tokens/s |
| Generation, 372-token response at 8k | 99.63 tokens/s |
| Time to first token, 372-token response at 8k | 0.551 s |
| Cold first request including model load at 8k | 8.73 s |
| Warm orchestrated response at 16k | 3.49 s |

The remaining VRAM margin is small. A vision model or GPU-resident embedding
model should not be loaded at the same time. For the RAG validation below,
the embedding model was deliberately placed on CPU.

## Test system

| Component | Value |
| --- | --- |
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU |
| VRAM reported by the runtime | 16,375 MiB |
| Driver | 595.79 |
| CUDA architecture | 8.9 / `sm_89` |
| CPU threads visible to the runtime | 32 |
| Threads used | 24 |
| Runtime build | `b1-fae3a28` |
| Test date | 2026-07-27 |

## Model

| Property | Value |
| --- | --- |
| Model | `google/gemma-4-26B-A4B-it` |
| GGUF | `gemma-4-26B_q4_0-it.gguf` |
| Quantization | Google QAT Q4_0 |
| File size | 14,439,363,584 bytes |
| SHA-256 | `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d` |
| Model context capacity | 262,144 tokens |
| Context used for this validation | 16,384 tokens |

The model file was downloaded from Google's
[official QAT Q4_0 GGUF repository](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf).

## Runtime configuration

The EIE router spawned a dedicated model process with the equivalent of:

```powershell
llama-server.exe `
  --model gemma-4-26B_q4_0-it.gguf `
  --alias google_gemma-4-26B-A4B-it-Q4_0 `
  --ctx-size 16384 `
  --cache-type-k q4_0 `
  --cache-type-v q4_0 `
  --flash-attn on `
  --n-gpu-layers all `
  --parallel 1
```

For latency measurements, Gemma's optional thinking trace was disabled in
the chat-template arguments. This avoids counting hidden deliberation tokens
as application latency:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

At 16k, the runtime reported:

```text
CUDA0 model buffer size = 13755.42 MiB
n_ctx = 16384
non-SWA KV buffer = 90.00 MiB (K q4_0: 45.00, V q4_0: 45.00)
SWA KV buffer = 84.38 MiB
```

## Raw EIE measurements

The most stable 16k samples used prompts of roughly 4,000 tokens:

| Prompt tokens | Prompt eval | Generated tokens | Generation |
| ---: | ---: | ---: | ---: |
| 3,957 | 2,613.96 tokens/s | 37 | 78.68 tokens/s |
| 4,016 | 2,441.87 tokens/s | 73 | 81.70 tokens/s |

An 8k long-generation sample produced 372 tokens in 4.285 seconds
end-to-end, with 0.551-second time to first token and 99.63 tokens/s
reported generation throughput.

## Orchestrated-client validation

A separate 12-task validation compared:

1. direct EIE inference with no tools, memory, RAG, or system message;
2. the same EIE-served model and sampling parameters behind a local
   orchestrator with file tools, time, an integrity ledger, and local RAG.

This section measures the operational value of an EIE client architecture.
It must not be interpreted as EIE making the underlying model more
intelligent.

| Condition | Total | Pure reasoning | Operational tasks | Mean latency | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct EIE | 6.33/12 | 6/6 | 0.33/6 | 11.548 s | 7.592 s |
| Orchestrated EIE client | 10.33/12 | 5/6 | 5.33/6 | 2.554 s | 2.206 s |

The direct condition was expected to fail tasks requiring unavailable
tools. The orchestrated condition made one code-tracing error. Its partially
scored memory task used an intentionally fresh, isolated database, so it did
not contain the historical beliefs requested by that task.

The RAG task was also repeated after the move to 16k context and CPU
embeddings. It retrieved the expected document passage and returned the
correct answer. The cold request after restart took 10.09 seconds; subsequent
warm responses took approximately 3.5 seconds.

## Practical guidance

- Use Q4 KV caches when fitting this model into 16 GB.
- Keep `--parallel 1`; additional slots would consume the remaining margin.
- Put embedding inference on CPU or another device.
- Do not co-reside a second substantial GPU model.
- Disable optional thinking for interactive latency measurements, or report
  hidden-token generation separately.
- Treat the 16k configuration as a high-utilization profile: leave more
  margin for unattended production workloads if other CUDA consumers are
  present.

The machine-readable summary is available in
[`gemma4-26b-a4b-rtx4090-laptop.json`](data/gemma4-26b-a4b-rtx4090-laptop.json).
