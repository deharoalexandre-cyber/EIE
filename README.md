# EIE: Elyne Inference Engine

**A local inference server for GGUF models: chat, embeddings and configurable multi-model execution.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-brightgreen.svg)](https://en.cppreference.com/w/cpp/17)

## In brief

- **What it is:** a C++ inference server that loads GGUF models locally and exposes HTTP chat and embedding endpoints using a subset of the OpenAI API format.
- **Who it is for:** developers of local assistants, RAG clients and desktop or self-hosted applications. **One LLM is enough; multiple models are optional.** [Elyne Next uses EIE/EWS daily at Elyne Corp](#production-use-elyne-next-resident-12b-and-streamed-26b).
- **What is verified:** bounded Windows/CUDA serving and EWS tests, Intel macOS serving, Windows x64 CPU serving on a consumer laptop, and Android CPU text chat on a Z Flip6 have local maintainer-run receipts. Apple Silicon/Metal and the older multi-device Android timings remain separately labelled *maintainer-reported*.
- **What is experimental:** EWS, an optional expert-weight streaming path for selected MoE models; GLM-5.3-Flash uses a separate experimental runtime.
- **Where to check:** [capability status](#capability-status), [claim-by-claim evidence](docs/CLAIMS_AUDIT.md) and [remaining work](docs/ROADMAP_TO_CLAIMS.md).

## What you can build with EIE

Start with a single GGUF model for a local chat application. Add an embedding model for a RAG client, or configure several loaded models for parallel, sequential or fan-out execution. The HTTP interface lets the client remain separate from model execution; [API compatibility is partial](#api-an-openai-shaped-subset-not-full-parity) and [group policies have documented limits](#scheduling-and-model-groups).

EIE provides inference, not the application itself: memory, document retrieval, tools, identity and agent orchestration belong to its clients. It is also distinct from the model weights it runs.

**EIE is the engine. EWS is one optional execution path.** Ordinary inference with a model that fits does not require expert streaming. EWS extends the engine to experiment with selected MoE models whose expert weights do not fit entirely in RAM or VRAM.

**Start here:** [Build](#build) → [Run one model](#quick-start) → [Call the API](#api-an-openai-shaped-subset-not-full-parity) → [Configure models and groups](#configuration). For streaming specifically: [Gemma EWS](docs/ews/runtime-port.md) · [GLM EWS](docs/GLM_EWS_EXPERIMENT.md).

The results below describe specific configurations, not requirements for using EIE.

## Production use: Elyne Next, resident 12B and streamed 26B

**Elyne Next is in production use at Elyne Corp: a resident Gemma 4 12B handles the ongoing conversation, with a Gemma 4 26B A4B auxiliary streamed through EIE/EWS for deeper analysis.**

The maintainer reports daily operation and public demonstrations at a trade show in September 2026; deployment to **two additional company workstations is in progress**. Here, **production means operational use, not a load or fleet-reliability qualification**. This working deployment is distinct from the experimental GLM replacement. The [claims audit](docs/CLAIMS_AUDIT.md) records the scope of technical validation separately from deployment reports.

Next's resident calls `request_deep_analysis`, supplies the question and context, then integrates the auxiliary's answer and continues the conversation. The 26B does not replace the resident or take over Next's memory and tools.

The [real application validation](docs/benchmarks/serving-functional-20260908.md#real-next-positive-route-and-a-retained-failure) records the complete **12B → streamed 26B → 12B** path: a 172-token auxiliary answer, resident integration and a subsequent resident turn. That profile uses a **16,384-token resident context**, a **4,096-token auxiliary context**, **F16 KV** and **16 EWS slots per layer**.

The [earlier coexistence measurements](docs/benchmarks/ews-consumed-20260905.md#real-next-not-just-an-empty-resident) separately report 26B output at **5.76–5.97 tokens per request-wall-second** alongside Next, with a sampled total GPU peak of **13,408 MiB**. Those timings include prefill and contention and used a 2,048-token auxiliary context; they are not throughput measurements of the later 4,096-token profile.

Next is a separate client application, not bundled in this repository. The GLM results below explore a larger auxiliary on a fresh Next copy; they do not redefine the working Gemma pair.

## Apple Silicon / Metal: running on a MacBook Pro

**EIE is running in the Elyne macOS application on a MacBook Pro (2021), with Apple Silicon, 16 GB unified memory and Gemma 4 E2B QAT Q4_0.** This is reported operation on a real machine, beyond a build recipe.

The Metal profile uses F16 KV and a 4,096-token context. With approximately **1,636 tokens of history already cached**, the engine-only trials report:

| Generated output | Reported time | Reported output rate |
|---|---|---|
| 512 tokens | 13.0 s measured, including 2.6 s queueing; about 10.5 s excluding that wait | 49 tok/s |
| 1,024 tokens | 21.7 s | 47 tok/s |
| 2,048 tokens | 46.4 s | 44 tok/s |

Time to first output in these warm-prefix trials is **0.07–0.10 s**. In actual application conversations, a 612-token answer took **22.3 s total**, including **14.3 s reported generation time**. Application timings also include memory, preparation and any web tools; they are not the engine-only figures above.

**Evidence status: maintainer-reported.** These on-device measurements are reported by the maintainer; no independent replication is claimed. They cover ordinary Metal inference, not Apple Silicon EWS or GLM qualification. See the [claims audit](docs/CLAIMS_AUDIT.md) for evidence scope. [Setup, full measurements and limits](docs/benchmarks/macos-apple-silicon-20260913.md) · [Build script](scripts/build-macos-arm64.sh) · [Metal preset](presets/macos-silicon.yaml).

## GLM/EWS milestone: GLM-5.3-Flash 320B on a laptop

**A real 12B → GLM-5.3-Flash → 12B roundtrip, with hybrid CPU/GPU expert-weight streaming. Locally validated by the maintainer on 8 September 2026.**

The quantized GLM artifact occupies **199.7 GB on disk** (six UD-Q4_K_XL shards). EWS does not keep every expert's weights in RAM or VRAM: it loads and reuses the experts selected by routing in bounded cache slots. On the tested laptop, GLM runs alongside Next's 12B resident.

| Latest measured configuration | Result |
|---|---|
| Hardware | RTX 4090 Laptop, **16 GiB-class VRAM / 32 GiB-class RAM** |
| Expert computation | **18 routed layers on GPU, 24 on CPU** |
| Actual application path | Next's 12B calls GLM, incorporates its answer, then completes another resident turn |
| Complete GLM answer | **148 tokens in 278.515 s - about 4 min 39 s** |
| Sampled GPU peak / remaining headroom | 14,995 MiB used / **1,054 MiB free**, resident included |

This is a **functional feasibility milestone**, not all-GPU inference, a speedup benchmark or a general quality guarantee. Numerical checks and Gemma non-regression tests accompany the result; their exact scope and the earlier truncated attempt are retained in the report.

**Read the evidence:** [GPU/Next report](docs/benchmarks/glm53-ews-gpu-next-20260908.md) · [Measurements, outputs and hashes](docs/benchmarks/data/glm53-ews-gpu-next-20260908.json).

**Reproduce GLM:** [Build the separate experimental runtime](docs/GLM_EWS_EXPERIMENT.md#reproduce-in-a-separate-checkout), then use the [hybrid GPU configuration and numerical checks](docs/GLM_EWS_EXPERIMENT.md#gpu-placement-reproduction). **The standard Gemma build below is a different runtime path.**

## What is established: updated 14 September 2026

The maintainers report single-model deployments in the Elyne lineage, sometimes with separate generation and embedding processes. This is not a public fleet-reliability study. EIE's group scheduler is implemented but not production/load-qualified.

The newer EWS path has **locally verified consumed-weight and Next-coexistence results**. Its source, runtime patch and measurement summaries are now published here. Local validation is evidence; it is not independent replication or a guarantee on every machine.

- [Claim-by-claim audit](docs/CLAIMS_AUDIT.md): code, evidence, qualifications.
- [Remaining work and acceptance criteria](docs/ROADMAP_TO_CLAIMS.md): what must still be built or measured.
- [Verification receipt](docs/benchmarks/claims-verification-20260907.md): checks actually performed, without another GPU campaign.
- [Serving validation receipt](docs/benchmarks/serving-functional-20260908.md): clean build, real 12B/26B and chat/embedding checks pass; Next's online 12B -> 26B -> 12B roundtrip succeeds.

**Labels:** *implemented* means the code path exists; *locally verified* names a bounded test executed by the maintainers, with a linked validation receipt; *maintainer-reported* describes operation or measurements reported by the maintainers without a complete published raw evidence bundle; *planned/not validated* is not a product guarantee. Neither evidence label implies independent replication. Historical reports retain their original results with explicit scope corrections; the [claims audit](docs/CLAIMS_AUDIT.md) details the evidence and its limits.

## Expert-Aware Weight Streaming (EWS)

**Consumed-weight path, locally validated by the maintainer on 5 September.** Expert matmuls use the slabs actually loaded into bounded per-layer slots. This is no longer just an I/O timing experiment.

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

**Experimental GLM EWS port, 8 September:** the [separate GLM runtime path](docs/GLM_EWS_EXPERIMENT.md) now streams the six-shard UD-Q4_K_XL artifact into an 8-slot host expert cache. The short native/EWS forward comparison passes: **619,520 logits bit-identical and the same four generated tokens**. Routed expert tensors occupy **5.152 GB instead of 185.478 GB**; that is the expert tensor allocation, not whole-process RAM or VRAM. This first profile computes experts on CPU, with partial CUDA offload for the rest of the model. The default Gemma runtime pin is unchanged.

The [actual fresh-state Next 12B -> GLM through EIE/EWS -> 12B roundtrip](docs/benchmarks/glm53-ews-next-20260908.md) also passes: a complete 702-token auxiliary answer in 1,090.547 s, resident synthesis, then a separate resident answer. Sampled total GPU use peaks at 12,894 MiB; whole-host RAM is tight. The answer exceeds the requested brevity and contains a corrected formatting error. This is a functional result, not a quality/speed claim; unsuccessful attempts are retained.

**Hybrid GLM GPU placement:** [numerical checks and a real Next roundtrip](docs/benchmarks/glm53-ews-gpu-next-20260908.md) now pass with actual CUDA expert buffers: a native/EWS control on two routed layers, then three cache-size consistency checks with 18 routed layers on GPU and 24 on CPU. This profile uses 2.198 GB of device expert cache plus 2.954 GB of host expert cache. The complete auxiliary answer is 148 tokens in 278.515 s, followed by resident synthesis and a new resident answer. Sampled total GPU peak is 14,995 MiB with 1,054 MiB free. A first 768-token truncated attempt is retained; the successful rerun used a 1,536-token allowance and a slightly different resident-authored question. This is **not** a paired speedup or a proven budget effect.

**Not validated for EWS:** other MoE architectures, arbitrary long contexts, multi-GPU EWS, Linux/ROCm/macOS/Android EWS performance, fleet reliability, power/water savings, or a 15–20% reduction in training GPUs. GLM all-GPU expert placement and full native equivalence at the larger GPU placement remain unqualified.

**Separate native GLM baseline:** a [native llama.cpp run](docs/benchmarks/glm53-native-next-20260908.md) already completed a fresh-state Next 12B -> GLM -> 12B roundtrip on the same laptop. Its 611.812 s auxiliary answer uses CPU expert mmap, **not EWS**. It is a working reference, not evidence that EWS is the only feasible path or a paired speed comparison.

## Desktop performance: historical field measurements

Maintainer-reported RTX 4090 Laptop / Windows 11 results. These are not a fresh benchmark of the current C++ server. E2B/E4B figures have no complete raw run bundle here. The [26B report](docs/benchmarks/gemma4-26b-a4b-rtx4090-laptop.md) identifies an earlier router plus `llama-server` runtime, with a configuration and JSON summary but no complete raw logs/prompts.

| Model / context | Quantization | GPU memory reported | Prompt eval | Decode |
|---|---|---|---|---|
| Gemma 4 E2B | Q6_K | ~2.5 GB | 3,146 t/s | 126 t/s |
| Gemma 4 E4B | Q6_K | ~4.5 GB | 1,883 t/s | 70 t/s |
| E2B + E4B loaded | Q6_K | ~7.5 GB | - | - |
| Gemma 4 26B A4B, 16k | QAT Q4_0 | 15,299–15,585 MiB | 2,442–2,614 t/s | 78.68–81.70 t/s |
| Same 26B, separate 8k sample | QAT Q4_0 | not separately reported | - | 99.63 t/s |

The old **“30% less VRAM / 2x faster than Ollama”** assertions are withdrawn as comparative claims: matched versions, context/cache settings and raw paired measurements are missing. Prefix reuse exists, but a prefix-similarity observation is not a throughput benchmark. The earlier three/six-model VRAM sizing table is also withdrawn as a sizing reference: complete model identities, memory accounting and measured peaks were missing.

## EIE Mobile (Android)

**Downloadable native CPU bundles:** [Android arm64 `dotprod` and `i8mm`](https://github.com/deharoalexandre-cyber/EIE/releases/tag/android-neon-2026.09.14), with EIE HTTP server, shared libraries, headers, licenses and SHA-256 manifests. [Run/rebuild guide](mobile/ANDROID_BUNDLE.md).

**Locally verified on 14 September:** both CPU variants run **Gemma 4 E2B QAT Q4_0** on a **Galaxy Z Flip6 (SM8650, Android 16)**. The release tests pass 628 native serving assertions per variant, matching buffered/SSE arithmetic answers and three 64-token generations per variant. [Receipt and limits](docs/benchmarks/android-neon-zflip6-20260914.md) / [per-request JSON, exact binaries and model hashes](docs/benchmarks/data/android-neon-zflip6-20260914.json). Raw dedicated ADB/server/HTTP outputs are downloadable with the release; earlier trials are retained too. These are CPU text-chat bundles, **not an APK, Android EWS, GPU/NPU or vision qualification**. Model weights are not included.

[`mobile/`](mobile/README.md) also contains the separate JNI wrapper and OpenCL/Adreno and Hexagon HTP build recipes. That integration still requires an application and externally supplied build dependencies; it is not validated by the new HTTP-bundle test. Prefix reuse avoids re-evaluating a retained identical prefix; it does not make the entire attention computation O(new tokens).

**Historical field measurements below, unchanged:** Gemma 4 E2B, QAT Q4_0 unless noted. These six rows still lack matched raw device/model/build bundles. They are maintainer-reported, not measurements of the new release. **No independent replication is claimed**; see the [claims audit](docs/CLAIMS_AUDIT.md).

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
| Explicit KV formats | KV format mapping implemented; F16 numerical EWS gate and Turbo3 API smoke, not an all-format quality benchmark |
| Automatic KV optimization | **Not operational**: no `auto` selector; health latency remains zero |
| Device-memory telemetry | Runtime port queries device memory; no reserve/budget/eviction enforcement |
| CUDA EWS | Local Windows/CUDA/Gemma validation, bounded above |
| Elyne Next: resident 12B + streamed 26B | [Daily operational use at Elyne Corp](#production-use-elyne-next-resident-12b-and-streamed-26b); the real tool call, auxiliary answer, resident integration and continuation are also [locally validated by the maintainer](docs/benchmarks/serving-functional-20260908.md#real-next-positive-route-and-a-retained-failure) |
| GLM EWS | Separate experimental runtime; host-cache and hybrid GPU-expert profiles have bounded numerical checks and actual fresh-state Next roundtrips; not all-GPU experts or general quality qualification |
| Apple Silicon / Metal | Reported real MacBook Pro operation with 16 GB unified memory and Gemma 4 E2B QAT Q4_0; warm-prefix engine-only output rates 44–49 tok/s, with separate application timings; [receipt](docs/benchmarks/macos-apple-silicon-20260913.md) |
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

The backend maps `f32`, `f16`, `q8_0`, `q4_0` and discovers `turbo2`, `turbo3`, `turbo4` by the runtime's KV type names. On the pinned Gemma fork these select `GGML_TYPE_TURBO*_0`, not the similarly named weight formats. The experimental native GLM fork has no TurboQuant KV types and uses explicit F16 in its measured profile. `turbo3` remains the default for the standard build, not a universal quality recommendation. Missing types or failed quantized context initialization may fall back to F16; check the logs.

On Apple Silicon, `turbo*` requests explicitly fall back to F16 with a logged warning: this fork's TurboQuant KV kernels are not available on Metal. The [measured Metal preset](presets/macos-silicon.yaml) selects F16 directly. Intel macOS remains a separate CPU-only profile.

There is **no `auto` selector**; unknown names fall back to F16. Separate K/V settings are available but are not qualified for every architecture. Quantizer bit widths do not equal whole-process VRAM savings.

`adaptKv()` can recreate a context while keeping weights loaded, but `health()` returns zero latency, so the advertised automatic downgrade does not trigger. Cache migration, output quality and latency bounds are not established.

`reserve_mb` is parsed but **not enforced**: the loader and request path do not call `VramManager::canLoad()`. Watermarks, `group_isolation` and group budgets are not parsed. Device-memory reporting does not implement these policies. Multiple aliases on one device must not have their device totals summed.

## Build

**Choose the runtime before building:**

- **Standard EIE / Gemma:** use the pinned submodule and `ews-runtime-2168b0.patch` in the commands below.
- **GLM-5.3-Flash 320B / EWS:** use the [separate-checkout recipe](docs/GLM_EWS_EXPERIMENT.md#reproduce-in-a-separate-checkout), with its GLM-specific runtime revision and patch. For the latest measured profile, follow the [hybrid GPU configuration](docs/GLM_EWS_EXPERIMENT.md#gpu-placement-reproduction). Do not stack the GLM patch on an already Gemma-patched runtime.

For the **standard path below**, initialize the pinned fork and apply its runtime patch **once**, before inference builds, including non-EWS profiles. The wrapper references fields added by that patch:

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
| Linux CPU (generic recipe) | `./scripts/build-cpu.sh` | Model/kernel compatibility and available RAM still apply; no receipt yet |
| Windows x64 CPU (consumer laptop) | [prebuilt bundle](docs/windows.md) or `scriptsuild-windows-cpu.bat` + `presets/windows-cpu.yaml` | Locally verified on a 2020 laptop (i7-10850H, 6 cores, 32 GB, no GPU used): [receipt](docs/benchmarks/windows-cpu-20260914.md) + [JSON](docs/benchmarks/data/windows-cpu-20260914.json); static GCC build, AVX2, no CUDA |
| macOS Intel | [prebuilt bundle](docs/macos.md) or `./scripts/build-macos-x86_64.sh` + `presets/macos-cpu.yaml` | Locally verified on a 2018 MacBook Pro (i5-8279U, 8 GB, CPU only): [receipt](docs/benchmarks/macos-intel-20260914.md) + [JSON](docs/benchmarks/data/macos-intel-20260914.json); Metal disabled by this project |
| Apple Silicon | [prebuilt bundle](docs/macos.md) or `./scripts/build-macos-arm64.sh` + `presets/macos-silicon.yaml` | Maintainer-reported MacBook Pro operation (JSON receipt from `scripts/receipt-macos.sh` pending): Metal, 16 GB memory, Gemma 4 E2B QAT Q4_0; 44–49 tok/s with a warm ~1.6k history, first output 0.07–0.10 s - [engine/application receipt](docs/benchmarks/macos-apple-silicon-20260913.md) |

The Windows EWS campaign does not qualify the portable reader, Metal, ROCm, or Android EWS. There is no “any OS / any GGUF” guarantee. Build duration depends on the machine.

## Quick start

**Start with one model that fits your machine.** After a successful build:

```bash
# Linux / macOS build directory
./build/eie-server -m model.gguf --ctx 8192 --port 8090
# Multiple explicitly loaded models
./build/eie-server -m model-a.gguf -m model-b.gguf --ctx 8192 --port 8090
```

On Windows, use `build-ews/Release/eie-server.exe` with its DLL directory on `PATH`, or the static CPU-only `eie-server.exe` from the [Windows bundle](docs/windows.md). This is normal loading; streaming requires an `ews_slots` setting as described in the [EWS guide](docs/ews/runtime-port.md).

**For GLM-5.3-Flash 320B streaming**, use the [separate GLM build and serving profile](docs/GLM_EWS_EXPERIMENT.md#reproduce-in-a-separate-checkout), then its [hybrid GPU settings](docs/GLM_EWS_EXPERIMENT.md#gpu-placement-reproduction). That experiment is not required for the single-model setup above.

`--models-dir` discovers files but **does not load them by itself**. Set `preload: [all]` in a preset only when all discovered models fit, or use repeatable `-m`. Other flags: `--config`/`-c`, `--host`, `--port`, `--ctx`.

## API: an OpenAI-shaped subset, not full parity

```bash
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"model","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Text messages use the GGUF's native chat template when available, with a generic fallback. EIE does not add an identity/persona; it disables the optional thinking channel in template rendering. Raw `prompt` passthrough is available when messages are not supplied.

**Compatibility limits:** native tool-call schemas/results, structured outputs, multimodal message arrays, seed handling and every SDK option are not implemented here. Unknown fields may be ignored. Serving now reports retained tokenized prompt length, sampled completion tokens and reused-prefix tokens in both modes. [Accounting semantics and qualification limits](tests/serving/README.md): serializer tests and the real-tokenizer gate pass on the identified 12B and streamed 26B profiles.

SSE and `one_shot` use the same incremental stop/UTF-8 path for streamed and buffered generation: supplying a stop no longer suppresses all callbacks. Offline, real-route/fake-model, native real-model and real HTTP chat/embedding tests pass in the recorded Windows/CUDA profiles. Long prompts are truncated by default; `truncate_prompt: false` requests an explicit overflow error (HTTP 400 with `context_length_exceeded` in nonstream mode). `strict_model: true` rejects unknown model IDs rather than accepting the single-model fallback. These are specific runtime options, not complete OpenAI compatibility.

| Endpoint | Method | Status |
|---|---|---|
| `/v1/chat/completions` | POST | Text chat, raw prompt, SSE; limits above |
| `/v1/embeddings` | POST | String/string-array input; encoder support depends on model; input is capped at 2,048 tokens |
| `/v1/models` | GET | Loaded-model registry |
| `/health` | GET | Process response/uptime and loaded-registry count, not inference readiness |
| `/v1/batch/execute` | POST | Configured group execution; not OpenAI Batch API |
| `/v1/chain/execute` | POST | Sequential chain |
| `/v1/admin/models/discover` | GET | Existing discovery registry, not a fresh scan |
| `/v1/admin/scheduling/status` | GET | Strategy name and group count |
| `/v1/admin/vram/status` | GET | Device memory per alias; shared-device values must not be summed |
| `/v1/admin/ews/status` | GET | EWS access/cache/read/payload counters |
| `/metrics` | GET | Prometheus-style text; synchronized metric maps and loaded-registry count; not full API load qualification |
| `/v1/admin/health/deep` | GET | Stub; no inference probe |
| `/v1/admin/config/reload` | POST | Stub |
| `/v1/completions`, admin load/unload | POST | Planned, not implemented |

### Known client integration issue

A [separate Next test with the 26B auxiliary stopped](docs/benchmarks/serving-functional-20260908.md) remains unresolved: the resident falsely attributes an answer to the auxiliary without calling the tool. This prevents full failure-handling qualification of that client workflow. The successful online GLM roundtrips are distinct tests; they neither fail because of this case nor establish that it is fixed.

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

For the current software project, cite the repository and identify the exact source revision you used. For a performance result, also cite its specific measurement report; the project citation alone does not establish a benchmark result.

```bibtex
@misc{deharo2026eie_software,
  author       = {De Haro, Alexandre},
  title        = {EIE: Elyne Inference Engine},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/deharoalexandre-cyber/EIE},
  note         = {Specify the source revision and, for performance claims, the measurement report used.}
}
```

<details>
<summary>Historical Zenodo archive: original title includes an adaptive-cache design objective, not an operational guarantee</summary>

The entry below preserves the archive's original bibliographic title and DOI. **Automatic KV adaptation is not operational in the current implementation.** This qualification is included in the BibTeX itself so it travels with the citation; the historical title is not the current feature summary.

```bibtex
@misc{deharo2026eie,
  author       = {De Haro, Alexandre},
  title        = {EIE: A Policy-Driven Multi-Model Inference Server with Adaptive KV Cache Compression and GPU-Agnostic Backend Abstraction},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19439972},
  url          = {https://doi.org/10.5281/zenodo.19439972},
  note         = {Historical archive title describes an adaptive-cache design objective; automatic KV adaptation is not operational in the current implementation.}
}
```

</details>
