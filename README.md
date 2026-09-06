# EIE — Elyne Inference Engine

**Local GGUF inference for one or several models, with experimental expert-weight streaming.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-brightgreen.svg)](https://en.cppreference.com/w/cpp/17)

EIE loads GGUF models and exposes chat and embedding endpoints using a subset of the OpenAI API format. It is inference infrastructure, not an agent: memory, tools, identity and application orchestration belong to its clients. **A single LLM is a supported use case; multiple models are not required.**

## What is established — 7 September 2026

The maintainers report single-model deployments in the Elyne lineage, sometimes with separate generation and embedding processes. This is not a public fleet-reliability study. EIE's group scheduler is implemented but not production/load-qualified.

The newer EWS path has **locally verified consumed-weight and Next-coexistence results**. Its source, runtime patch and measurement summaries are now published here. Local validation is evidence; it is not independent replication or a guarantee on every machine.

- [Claim-by-claim audit](docs/CLAIMS_AUDIT.md): code, evidence, qualifications.
- [Remaining work and acceptance criteria](docs/ROADMAP_TO_CLAIMS.md): what must still be built or measured.
- [Verification receipt](docs/benchmarks/claims-verification-20260907.md): checks actually performed, without another GPU campaign.

**Labels:** *implemented* means the code path exists; *locally verified* names a bounded executed test; *maintainer-reported* lacks a complete inspected raw evidence bundle; *planned/not validated* is not a product guarantee. Historical reports retain their original results with explicit scope corrections.

## Expert-Aware Weight Streaming (EWS)

**Consumed-weight path, locally validated on 5 September.** Expert matmuls use the slabs actually loaded into bounded per-layer slots. This is no longer just an I/O timing experiment.

| Local result | Evidence and limits |
|---|---|
| 26B CUDA weight buffers: **14.424 GB → 4.789 GB**, 32 slots/layer | Pilot; decimal GB, weights only, not total VRAM/KV |
| **14.974 GB** expert payload transferred during 95 decode calls | French pilot, 32 slots; 4,476 misses, actual evictions and consumed weights |
| Five positive pilot comparisons have bit-identical logits; zero-weight control diverges | Two short prompts, 96 predictions; matched instrumentation, not all-input equivalence |
| EIE runtime port: four bit-identical candidate/reference comparisons | Two prompts, 32 predictions; F16/CUDA, 32 or 8 slots/layer |
| Real Next 12B active alongside streamed 26B | Two five-turn runs on copied state; 26B 16 slots, 2k context; Next configured 16k, observed prompts up to 9,494 tokens |
| 26B **5.76–5.97 output tokens/request-wall-second** alongside Next | Includes prefill and contention, not isolated decode; sampled peak GPU use 13,408 MiB |

The [September report](docs/benchmarks/ews-consumed-20260905.md) links the evidence and its local re-verification. [Build/run instructions](docs/ews/runtime-port.md) require the supplied patch to the pinned llama.cpp submodule. The source includes the local runtime port and the 6 September SSE cancellation fix; model weights, binaries and Next itself are not bundled.

**Do not combine different experiments.** The [August C+ archive](docs/benchmarks/ews-gemma4-a4b-rtx4090-laptop.md) reported 6.55–11.10 tok/s with real I/O inserted during decode, but its FFN still consumed resident weights. Those timings exclude prefill/loading and do not prove consumed streaming or physical VRAM savings. Its hotset/SLRU and first-read SHA scheme are **not** the policy of the September runtime, which uses an LRU-style slot cache without per-chunk SHA verification.

**Not validated:** GLM 320B or other MoE architectures, arbitrary long contexts, multi-GPU EWS, Linux/ROCm/macOS/Android EWS performance, fleet reliability, power/water savings, or a 15–20% reduction in training GPUs. These are research or qualification targets, not consequences of the Gemma pilot.

## Desktop performance — historical field measurements

Maintainer-reported RTX 4090 Laptop / Windows 11 results. These are not a fresh benchmark of the current C++ server. E2B/E4B figures have no complete raw run bundle here. The [26B report](docs/benchmarks/gemma4-26b-a4b-rtx4090-laptop.md) identifies an earlier router plus `llama-server` runtime, with a configuration and JSON summary but no complete raw logs/prompts.

| Model / context | Quantization | GPU memory reported | Prompt eval | Decode |
|---|---|---|---|---|
| Gemma 4 E2B | Q6_K | ~2.5 GB | 3,146 t/s | 126 t/s |
| Gemma 4 E4B | Q6_K | ~4.5 GB | 1,883 t/s | 70 t/s |
| E2B + E4B loaded | Q6_K | ~7.5 GB | — | — |
| Gemma 4 26B A4B, 16k | QAT Q4_0 | 15,299–15,585 MiB | 2,442–2,614 t/s | 78.68–81.70 t/s |
| Same 26B, separate 8k sample | QAT Q4_0 | not separately reported | — | 99.63 t/s |

The old **“30% less VRAM / 2x faster than Ollama”** assertions are withdrawn as comparative claims: matched versions, context/cache settings and raw paired measurements are missing. Prefix reuse exists, but a prefix-similarity observation is not a throughput benchmark. The earlier three/six-model VRAM sizing table is also withdrawn as a sizing reference: complete model identities, memory accounting and measured peaks were missing.

## EIE Mobile (Android)

[`mobile/`](mobile/README.md) contains a JNI wrapper and Android arm64 build recipes for CPU variants, OpenCL/Adreno and Hexagon HTP. This is **not a ready-to-build APK**: the application, prebuilt libraries, headers and external build paths must be supplied. Prefix reuse avoids re-evaluating a retained identical prefix; it does not make the entire attention computation O(new tokens).

Maintainer-reported on-device results (Gemma 4 E2B, QAT Q4_0 unless noted). No raw device logs, complete model/build hashes or reproducible app bundle are included; these numbers were **not independently verified in this audit**.

| SoC | Backend | Decode | Prompt eval |
|---|---|---|---|
| Snapdragon 8 Elite | HTP v79 | 19.6–20.6 t/s | 860–959 t/s |
| Snapdragon 8 Gen 3 | HTP v75 | 13.4–17.8 t/s | 600–664 t/s |
| Snapdragon 8 Gen 3 | CPU i8mm, Q4_K_M | 17.4–18.7 t/s | 43–53 t/s |
| Snapdragon 8 Gen 3 | Adreno 750, Q4_0 | 8.0–9.7 t/s | 204–229 t/s |
| Snapdragon 888 | CPU dotprod | 7.0–8.7 t/s | ~31 t/s |
| Dimensity 9000+ | CPU i8mm | ~6 t/s | ~27 t/s |

Bandwidth and compute limitations are possible explanations, not causal conclusions established by this table. Different quantizations and missing matched configurations prevent a general CPU/GPU/NPU speedup claim. See the mobile README for device and packaging limitations.

## Capability status

This replaces an unversioned competitor comparison. Absence or inferiority of features in other engines was not established by this repository.

| Capability | Current status |
|---|---|
| One-model chat / embeddings | Implemented; local use reported; API limits below |
| Parallel / sequential / fan-out groups | Implemented; basic API smoke tests, not full group acceptance/load qualification |
| `strict` / `partial` outcomes | Implemented; `retry_once` does not retry, `replace_with` does not invoke a replacement |
| Explicit KV formats | Mapping corrected in this publication; F16 numerical EWS gate and Turbo3 API smoke, not an all-format quality benchmark |
| Automatic KV optimization | **Not operational**: no `auto` selector; health latency remains zero |
| Device-memory telemetry | Runtime port queries device memory; no reserve/budget/eviction enforcement |
| CUDA EWS | Local Windows/CUDA/Gemma validation, bounded above |
| Other platforms | Build paths or portable code; not qualified by the Windows campaign |
| Audit logging | Optional FNV-derived prototype, not a cryptographic/verifiable audit ledger |
| Custom strategy plugins | Interface exists; dynamic library loading planned |

### Scheduling and model groups

| Strategy | Current behavior |
|---|---|
| `generic` | Boot-time loading; per-model mutex serializes inference, without FIFO/fairness guarantee; on-demand loading/eviction planned |
| `pinned-group` | Boot-loaded members and response quorum; not a separate enforced memory reservation |
| `multi-group` | Alias of `pinned-group`; distinct multi-group policy still planned |
| `fixed-appliance` | Boot-loaded models, partial-result policy; no dynamic loader |

Parallel execution sends the prompt to multiple loaded backends using asynchronous calls. Sequential execution makes each response the next model's input. Fan-out selects the **longest successful response**, not the best-quality response. A threshold in `max_latency_ms` is not an execution timeout.

`retry_once` currently returns a partial outcome after one failed call; `replace_with` fails without invoking the replacement. Group KV overrides are not parsed, and their non-empty struct defaults can mask global KV/context settings. These are [known unfinished behaviors](docs/ROADMAP_TO_CLAIMS.md), not advertised recovery guarantees.

### KV cache and memory management

The backend maps `f32`, `f16`, `q8_0`, `q4_0`, `turbo2`, `turbo3`, `turbo4`. This publication corrects the old `turbo*` mapping, which selected weight formats instead of the fork's `GGML_TYPE_TURBO*_0` KV formats. `turbo3` is the configured default, not a universal quality recommendation. A failed quantized context initialization may fall back to F16; check the logs.

There is **no `auto` selector**; unknown names fall back to F16. Separate K/V settings are available but are not qualified for every architecture. Quantizer bit widths do not equal whole-process VRAM savings.

`adaptKv()` can recreate a context while keeping weights loaded, but `health()` returns zero latency, so the advertised automatic downgrade does not trigger. Cache migration, output quality and latency bounds are not established.

`reserve_mb` is parsed but **not enforced**: the loader and request path do not call `VramManager::canLoad()`. Watermarks, `group_isolation` and group budgets are not parsed. Device-memory reporting does not implement these policies. Multiple aliases on one device must not have their device totals summed.

## Build

Initialize the pinned fork and apply the supplied runtime patch **once**, before any inference build, including non-EWS profiles. The wrapper references fields added by that patch:

```bash
git clone https://github.com/deharoalexandre-cyber/EIE.git
cd EIE
git submodule update --init
git -C llama.cpp apply ../patches/ews-runtime-2168b0.patch
```

Do not reapply it to an already-patched checkout. Without the submodule, CMake builds a placeholder demonstration, **not** an HTTP inference server. The [EWS build recipe](docs/ews/runtime-port.md) records the locally tested Windows/CUDA profile and dependency requirements.

### Windows (CUDA)

Requires Visual Studio 2022 Build Tools with Desktop C++, CMake and CUDA. The local EWS campaign used CUDA 13.2 / MSVC 19.44. From a Developer PowerShell after the setup above:

```powershell
cmake -B build-ews -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_MTMD=OFF -DLLAMA_OPENSSL=OFF -DEIE_BUILD_EWS_TESTS=ON
cmake --build build-ews --config Release --target eie-server ews-forward -j 6
```

`89` is the tested GPU architecture, not a universal setting. The binary is `build-ews/Release/eie-server.exe`; its runtime DLL directory is `build-ews/bin/Release`. See the runtime guide for `PATH` and CUDA-graph settings. No prebuilt executable is distributed here.

### Linux and macOS build targets

After the same submodule/patch setup, recipes are available:

| Target | Command | Qualification |
|---|---|---|
| Linux NVIDIA | `./scripts/build-cuda.sh` | Prior Linux operation reported; no complete published native run bundle |
| Linux AMD | `./scripts/build-rocm.sh` | ROCm target, not a validated first-class device matrix |
| CPU | `./scripts/build-cpu.sh` | Model/kernel compatibility and available RAM still apply |
| macOS 15 Intel | CPU recipe + `presets/macos-cpu.yaml` | Maintainer-reported operation; Metal disabled by this project |
| Apple Silicon | `./scripts/build-macos-arm64.sh` | arm64/Metal build recipe; native qualification still required |

The Windows EWS campaign does not qualify the portable reader, Metal, ROCm, or Android EWS. There is no “any OS / any GGUF” guarantee. Build duration depends on the machine.

## Quick start

For a model that fits, after a successful build:

```bash
# Linux / macOS build directory
./build/eie-server -m model.gguf --ctx 8192 --port 8090
# Multiple explicitly loaded models
./build/eie-server -m model-a.gguf -m model-b.gguf --ctx 8192 --port 8090
```

On Windows, use `build-ews/Release/eie-server.exe` with its DLL directory on `PATH`. This is normal loading; streaming requires an `ews_slots` setting as described in the [EWS guide](docs/ews/runtime-port.md).

`--models-dir` discovers files but **does not load them by itself**. Set `preload: [all]` in a preset only when all discovered models fit, or use repeatable `-m`. Other flags: `--config`/`-c`, `--host`, `--port`, `--ctx`.

## API — an OpenAI-shaped subset, not full parity

```bash
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"model","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Text messages use the GGUF's native chat template when available, with a generic fallback. EIE does not add an identity/persona; it disables the optional thinking channel in template rendering. Raw `prompt` passthrough is available when messages are not supplied.

**Compatibility limits:** native tool-call schemas/results, structured outputs, multimodal message arrays, seed handling, complete usage accounting and every SDK option are not implemented here. Unknown fields may be ignored. `prompt_tokens` remains zero; do not use it for billing or context validation.

SSE and `one_shot` have local regression evidence. `stop` combined with `stream=true` is **not qualified**: token callbacks are suppressed when stop sequences are supplied. Long prompts are truncated by default; `truncate_prompt: false` requests an explicit overflow error. `strict_model: true` rejects unknown model IDs rather than accepting the single-model fallback. These are specific runtime options, not complete OpenAI compatibility.

| Endpoint | Method | Status |
|---|---|---|
| `/v1/chat/completions` | POST | Text chat, raw prompt, SSE; limits above |
| `/v1/embeddings` | POST | String/string-array input; encoder support depends on model; input is capped at 2,048 tokens |
| `/v1/models` | GET | Loaded-model registry |
| `/health` | GET | Process response/uptime, not inference readiness; model count is activity-derived |
| `/v1/batch/execute` | POST | Configured group execution; not OpenAI Batch API |
| `/v1/chain/execute` | POST | Sequential chain |
| `/v1/admin/models/discover` | GET | Existing discovery registry, not a fresh scan |
| `/v1/admin/scheduling/status` | GET | Strategy name and group count |
| `/v1/admin/vram/status` | GET | Device memory per alias; shared-device values must not be summed |
| `/v1/admin/ews/status` | GET | EWS access/cache/read/payload counters |
| `/metrics` | GET | Prometheus-style text; concurrency and loaded-count limitations remain |
| `/v1/admin/health/deep` | GET | Stub; no inference probe |
| `/v1/admin/config/reload` | POST | Stub |
| `/v1/completions`, admin load/unload | POST | Planned, not implemented |

## Configuration

The reader accepts a minimal YAML subset, not general YAML. See `presets/`. A single-model example:

```yaml
host: 127.0.0.1
port: 8090
strategy: generic
auto_discover: false
type_k: f16
type_v: f16
flash_attn: true
n_ctx: 8192
models:
  model: /models/model.gguf
preload: [model]
```

Groups use `name`, `models`, `required_responses`, `type`, `pinned`, `fallback` and `max_latency_ms`; their limitations are documented above. `dual-core-six.yaml` is a strategy skeleton, **not** a ready six-model configuration.

`auth_token` is parsed but is not checked by HTTP handlers; it provides no authentication. Choose network exposure accordingly. The optional audit prototype uses an FNV-derived digest, omits inputs needed to reconstruct it, and resets its chain on restart. It is **not** a cryptographic tamper-evident or append-only integrity guarantee. These qualifications do not add a security layer to the runtime.

## Docker and migration

The Dockerfiles are **unqualified recipes**, not deployment-ready images: they do not apply the required runtime patch and copy only the executable, which may omit shared libraries. Clean-container inference and image digests remain to be supplied.

For migration from another engine, use a compatible GGUF artifact, point your client at EIE's `/v1` base URL, and test the specific fields and behavior it uses. No universal drop-in compatibility, speed advantage or memory reduction is established.

## Contributing, attribution and license

See [CONTRIBUTING.md](docs/CONTRIBUTING.md). Extension interfaces exist; dynamic strategy plugins remain planned.

EIE's code is [Apache 2.0](LICENSE), copyright 2026 Elyne Corp. The [llama.cpp fork](https://github.com/TheTom/llama-cpp-turboquant), dependencies, toolchain components and model weights retain their own licenses. See [NOTICE](NOTICE). This is not an Apache-2.0 relicensing of those artifacts.

Acknowledgments: [llama.cpp](https://github.com/ggml-org/llama.cpp), [TheTom's TurboQuant work](https://github.com/TheTom/llama-cpp-turboquant), and [AirLLM](https://github.com/lyogavin/airllm) as an inspiration for weight streaming. No competitor performance ranking is implied.

## Citation

The historical archived title includes an adaptive-cache design objective; it is not evidence that automatic KV adaptation is operational. Cite a report and revision for performance claims.

```bibtex
@misc{deharo2026eie,
  author       = {De Haro, Alexandre},
  title        = {EIE: A Policy-Driven Multi-Model Inference Server with Adaptive KV Cache Compression and GPU-Agnostic Backend Abstraction},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19439972},
  url          = {https://doi.org/10.5281/zenodo.19439972}
}
```
