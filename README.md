# EIE — Elyne Inference Engine

**A generic, policy-driven, multi-model GGUF inference server.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-brightgreen.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ROCm](https://img.shields.io/badge/AMD-ROCm-ED1C24.svg)](https://www.amd.com/en/products/software/rocm.html)
[![Windows](https://img.shields.io/badge/Windows-CUDA%2013.2-0078D6.svg)](#windows-cuda)

---

EIE is a local inference server that loads GGUF models, serves them via an OpenAI-compatible REST API, and manages GPU memory. It is designed as **infrastructure** — it serves completions, nothing more. Orchestrators, agents, and domain-specific logic are clients of this server.

**Production status (September 2026).** EIE runs the Elyne lineage in **single-model** mode (one generation model per instance, plus a separate embedding instance). Elyne Core Origin converges **two** LLMs through two dedicated single-model server processes orchestrated by the application, not through EIE model groups. The group scheduler below (parallel / sequential / fan-out) is **implemented and API-tested, but not yet exercised in production nor load-benchmarked**. Two-model co-residency is measured (Gemma 4 E2B + E4B, ~7.5 GB); larger groups are projections. Each feature table in this document states what is measured, what is implemented, and what is planned — we describe what ships.

EIE also ships an **embedded Android flavor** — same TurboQuant fork compiled for arm64,
with runtime CPU-variant dispatch (ARMv8.2 → v9), OpenCL/Adreno and Hexagon NPU (HTP)
backends, and KV-reuse incremental generation. See [`mobile/`](mobile/README.md).

## Performance

Benchmarked on NVIDIA GeForce RTX 4090 Laptop GPU (16 GB VRAM), Windows 11:

| Model | Quant | VRAM | Prompt eval | Generation |
|---|---|---|---|---|
| Gemma 4 E2B | Q6_K | ~2.5 GB | 3,146 t/s | 126 t/s |
| Gemma 4 E4B | Q6_K | ~4.5 GB | 1,883 t/s | 70 t/s |
| Gemma 4 26B A4B | QAT Q4_0 | 15.3-15.6 GB | 2,442-2,614 t/s | 79-100 t/s |
| **Both loaded** | Q6_K | **~7.5 GB** | — | — |

The 26B A4B result uses a 16,384-token context with Q4_0 KV caches on a
16 GB RTX 4090 Laptop GPU. See the
[full configuration and field report](docs/benchmarks/gemma4-26b-a4b-rtx4090-laptop.md).

Compared to Ollama with the same models on the same hardware:
- **30% less VRAM** (7.5 GB vs 10.8 GB)
- **2x faster generation** (126 t/s vs ~60 t/s on E2B)
- Prompt cache active: `sim_best = 0.876` — subsequent requests are faster

## EIE Mobile (Android)

The same engine, embedded. [`mobile/`](mobile/README.md) ships a JNI wrapper and Docker
build recipes that put the TurboQuant fork on Android arm64 — one APK covering ARMv8.2
(Snapdragon 888) through ARMv9, with the best CPU variant selected at runtime, plus
OpenCL/Adreno and Hexagon NPU (HTP) offload, and KV-reuse incremental generation
(O(new tokens) instead of O(history) per turn).

Measured on-device (Gemma 4 E2B, QAT Q4_0 unless noted):

| SoC | Backend | Generation | Prompt eval |
|---|---|---|---|
| Snapdragon 8 Elite | Hexagon HTP v79 | **19.6–20.6 t/s** | **860–959 t/s** |
| Snapdragon 8 Gen 3 | Hexagon HTP v75 | 13.4–17.8 t/s | 600–664 t/s |
| Snapdragon 8 Gen 3 | CPU i8mm (Q4_K_M) | 17.4–18.7 t/s | 43–53 t/s |
| Snapdragon 8 Gen 3 | Adreno 750 (Q4_0) | 8.0–9.7 t/s | 204–229 t/s |
| Snapdragon 888 | CPU dotprod variant | 7.0–8.7 t/s | ~31 t/s |
| Dimensity 9000+ | CPU i8mm | ~6 t/s | ~27 t/s |

On phones, generation is DRAM-bandwidth-bound while prompt eval is compute-bound:
matrix accelerators win prompt eval 4–20×, and QAT Q4_0 + HTP repack shrink the
per-token working set enough that the NPU also wins generation on recent SoCs.
Constraints, known limitations and build recipes: [`mobile/README.md`](mobile/README.md).

## Experimental: Expert-Aware Weight Streaming (EWS)

Streaming MoE expert weights from NVMe at expert granularity, with a
calibrated hotset + SLRU cache and fail-closed per-chunk SHA-256 verification.
A frozen v1 candidate passed a pre-registered end-to-end gate on **four unseen
holdout workloads**:

> Gemma 4 26B-A4B Q4_0 — 4/4 unseen holdouts passed —
> **6.55–11.10 tok/s end-to-end** — **40–48% fewer cold bytes/token vs static
> hotset** — SHA-256 chunk verification fail-closed enabled.
> Scope: Gemma-4-A4B, n_ctx ≤ 4096, tested hardware only.
> *No claim is made for K3-class models, other MoE architectures, or contexts
> above 4096 tokens.*

The campaign includes two pre-registered negative results (Mixtral-class
coarse MoE killed as a target; a first cache policy rejected on holdouts) —
kept published because they define the eligibility rule: *EWS eligibility is
governed by the cold working set induced by routing, not by model size.*
Design: [docs/ews/README.md](docs/ews/README.md) · Field report:
[docs/benchmarks/ews-gemma4-a4b-rtx4090-laptop.md](docs/benchmarks/ews-gemma4-a4b-rtx4090-laptop.md)

## Why EIE?

|  | Ollama | vLLM | llama.cpp server | **EIE** |
| --- | --- | --- | --- | --- |
| **Scheduling** | None (FIFO) | Continuous batching | None (FIFO) | Policy-driven (pluggable) |
| **Model Groups** | No | No | No | Parallel, Sequential, Fan-out (implemented, API-tested; not load-benchmarked) |
| **Fallback** | No | No | No | strict / partial / retry / replace |
| **KV Cache** | f16 / q8 / q4 | f16 / FP8 | f16 / q8 / q4 + TurboQuant | All legacy + **TurboQuant turbo2/3/4** |
| **Adaptive KV** | No | No | No | Health-check → auto downgrade |
| **Multi-model** | Sequential (swap) | Single model | Single model | **Simultaneous** (two models measured co-resident; eviction under VRAM pressure planned) |
| **Windows** | Yes | No | Yes | **Yes (CUDA 13.2 validated)** |
| **On-device (Android)** | No | No | CLI via Termux | **Embedded engine (CPU / Adreno / Hexagon NPU)** |
| **NVIDIA** | CUDA | CUDA | CUDA | CUDA (native) |
| **AMD** | Experimental | Partial | Partial | **ROCm first-class** |
| **VRAM mgmt** | Opaque | Per-request | None | Reserve + watermarks parsed; enforcement and per-group budgets planned |
| **Audit** | No | No | No | Hash-chained audit trail |
| **License** | MIT | Apache 2.0 | MIT | **Apache 2.0** |

## Key Features

### Policy Engine (pluggable)

Scheduling behavior is defined by strategies, not hardcoded. Four built-in strategies ship with EIE. Loading custom strategies from shared libraries (`plugin:libcustom.so`) is planned but not implemented yet.

```yaml
strategy: pinned-group          # or: generic, multi-group, fixed-appliance
```

| Strategy | Behavior | Use case |
| --- | --- | --- |
| `generic` | Boot-time loading, FIFO; on-demand loading and LRU eviction **planned** (policy hooks exist, not wired) | Ollama replacement (partial today) |
| `pinned-group` | N models pinned, multi-response required | Multi-model deliberation |
| `multi-group` | Today an alias of `pinned-group`; distinct per-group rules **planned** | Dual-core architectures (target) |
| `fixed-appliance` | Pre-loaded at boot, no dynamic loading | Embedded / edge devices |

### Model Groups

The scheduler operates on **groups**, not individual models. A group is a set of models with an execution rule.

```yaml
groups:
  - name: core
    models: [model-a, model-b, model-c]
    required_responses: 3
    type: parallel          # parallel / sequential / fanout
    pinned: true
    fallback: partial       # strict / partial / retry_once / replace_with
```

**Parallel** — Same prompt to N models simultaneously. All responses returned.
**Sequential** — Output of model N becomes input of model N+1 (pipeline).
**Fan-out** — Same prompt to N models; the **longest** successful response is selected (placeholder heuristic — quality-based selection planned).

### Adaptive KV Cache (TurboQuant)

TurboQuant KV cache compression is a **first-class capability**, not an afterthought. EIE supports all formats with an `auto` mode that selects the optimal compression based on available VRAM.

| Mode | Bits/value | Compression | When to use |
| --- | --- | --- | --- |
| `f16` | 16 | 1x | Debug, baseline |
| `q8_0` | 8 | ~2x | Sensitive K precision |
| `turbo4` | 4 | ~4x | Quality > compression |
| **`turbo3`** | **3.5** | **~5x** | **Production default** |
| `turbo2` | 2 | ~6.4x | Extreme memory pressure |
| asymmetric | K:8 / V:3.5 | K:2x / V:5x | Sensitive models |

The health-check mechanism can trigger **runtime KV downgrade** (e.g., turbo3 → turbo2) without reloading the model, keeping group execution within latency bounds.

### VRAM Quality of Service

> **Status:** `reserve_mb` is honored at load time. Watermarks, `group_isolation` and per-group budgets are parsed but **not enforced yet** — the VRAM manager is not wired into the request path. Read the block below as the target contract, not current behavior.

```yaml
vram:
  reserve_mb: 512           # always keep free
  low_watermark: 85         # start evicting non-pinned
  critical_watermark: 95    # force eviction
  group_isolation: true     # per-group VRAM budgets
```

### Compute Backend Abstraction

One codebase. The backend is selected at build time (`GGML_CUDA` / `GGML_HIP`, otherwise CPU) and initialized at runtime; all three wrap the same llama.cpp fork.

## Build

### Linux (CUDA)

```bash
git clone https://github.com/deharoalexandre-cyber/EIE.git
cd EIE && git submodule update --init
./scripts/build-cuda.sh
./build/eie-server --config presets/generic.yaml
```

### Linux (ROCm / AMD)

```bash
git submodule update --init
./scripts/build-rocm.sh
```

### Linux (CPU only)

```bash
git submodule update --init
./scripts/build-cpu.sh
```

### macOS (CPU)

Works on Intel and Apple Silicon Macs. Requires CMake (`brew install cmake`).

```bash
git submodule update --init
./scripts/build-cpu.sh
./build/eie-server --config presets/macos-cpu.yaml
```

> **Note:** On Intel Macs, Metal is disabled at configure time — non-Apple-Silicon
> GPUs produce incorrect results with the Metal backend. Inference runs on CPU.

### Windows (CUDA)

**Prerequisites:**
- [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/) — with **Desktop development with C++** workload
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (12.x or 13.x)
- [CMake](https://cmake.org/download/) (3.14+)

**Automated build:**

Open **Developer PowerShell for VS 2022** and run:

```powershell
git clone https://github.com/deharoalexandre-cyber/EIE.git
cd EIE
git submodule update --init
.\scripts\build-windows-cuda.bat
```

**Manual build:**

```powershell
# Set CUDA environment
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:CudaToolkitDir = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"

# Fix ASM issue (if MASM not installed)
(Get-Content 'llama.cpp\ggml\CMakeLists.txt') -replace 'project\(ggml C CXX ASM\)', 'project(ggml C CXX)' | Set-Content 'llama.cpp\ggml\CMakeLists.txt'

# Fix MSVC storage-class error (GGML_API already carries `extern` on MSVC)
(Get-Content 'llama.cpp\ggml\src\ggml-cpu\ops.cpp') -replace 'GGML_API extern int turbo3_cpu_wht_group_size', 'extern int turbo3_cpu_wht_group_size' | Set-Content 'llama.cpp\ggml\src\ggml-cpu\ops.cpp'

# Configure and build (static link — required by the extern fix above)
cmake -B build -G "Visual Studio 17 2022" -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCUDAToolkit_ROOT="$env:CUDA_PATH" -DCMAKE_CUDA_COMPILER="$env:CUDA_PATH\bin\nvcc.exe"
cmake --build build --config Release
```

Both fixes are applied automatically by `scripts\build-windows-cuda.bat`.

> **Note:** Build takes 15-20 minutes due to CUDA kernel compilation. The binary is at `build\Release\eie-server.exe`.

## Quick Start

### Single model

```bash
# Linux / macOS
./build/eie-server -m model.gguf --ctx 8192 --port 8090

# Windows
build\Release\eie-server.exe -m model.gguf --ctx 8192 --port 8090
```

### Multi-model router

Load all GGUF models from a directory and route by model name via the API:

```bash
# Linux / macOS
./build/eie-server --models-dir /path/to/models --ctx 8192 --port 8090

# Windows
build\Release\eie-server.exe --models-dir C:\Users\User\models --ctx 8192 --port 8090
```

CLI flags: `--config`/`-c <path>` (YAML preset), `-m <model.gguf>` (repeatable),
`--models-dir <dir>`, `--host`, `--port`, `--ctx <n>`. GPU offload is automatic
when built with CUDA/ROCm.

## API

### Layer 1 — OpenAI Compatible (drop-in)

Any OpenAI-compatible client works without modification.

```bash
# Chat completion
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google_gemma-4-E4B-it-Q6_K",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# List models
curl http://localhost:8090/v1/models
```

**Native chat template.** `messages[]` is rendered with the model's **own chat
template**, read from the GGUF metadata (`<start_of_turn>` for Gemma, ChatML for
Qwen, etc.). EIE never injects a persona or extra prompt text — the substrate
receives the conversation in its native syntax, nothing more. Models without a
usable template fall back to a generic `User:/Assistant:` layout.

Two escape hatches:
- `"prompt"` (string) — raw completion passthrough, no template applied. Full
  control for clients that assemble their own context.
- `"stop"` (string or array) — cut generation on custom sequences, in addition
  to the model's native end-of-generation token.

Group endpoints (`/v1/batch/execute`, `/v1/chain/execute`) apply the same rule
per model: each backend in the group renders the shared `messages[]` with the
native template of **its own** GGUF before generating.

| Endpoint | Method | Status | Description |
| --- | --- | --- | --- |
| `/v1/chat/completions` | POST | ✅ | Chat completion (native template, `prompt` passthrough, `stop`) |
| `/v1/models` | GET | ✅ | List loaded models |
| `/health` | GET | ✅ | Server health |
| `/v1/embeddings` | POST | ✅ | Embeddings (encoder models, e.g. bge-m3) |
| `/v1/completions` | POST | 🚧 planned | Text completion |

### Layer 2 — Generic Extensions

| Endpoint | Method | Status | Description |
| --- | --- | --- | --- |
| `/v1/batch/execute` | POST | ✅ API-tested, not load-benchmarked | Execute a model group (N parallel responses, per-model native template) |
| `/v1/chain/execute` | POST | ✅ API-tested, not load-benchmarked | Execute a sequential chain (pipeline) |
| `/v1/admin/models/discover` | GET | ✅ | Scan model directory |
| `/v1/admin/scheduling/status` | GET | ✅ | Active policy and groups |
| `/metrics` | GET | ✅ | Prometheus-compatible metrics |
| `/v1/admin/vram/status` | GET | 🚧 stub | VRAM per GPU, per model, per group |
| `/v1/admin/config/reload` | POST | 🚧 stub | Hot-reload YAML configuration |
| `/v1/admin/models/load` | POST | 🚧 planned | Load a GGUF model into VRAM |
| `/v1/admin/models/unload` | POST | 🚧 planned | Unload a model |

#### Group Execution

```bash
# Execute a 3-model group in parallel
curl http://localhost:8090/v1/batch/execute \
  -H "Content-Type: application/json" \
  -d '{
    "group": "core",
    "messages": [{"role": "user", "content": "Analyze this alert"}]
  }'
```

## Configuration

```yaml
# /etc/eie/engine.yaml
host: 0.0.0.0
port: 8090
strategy: pinned-group
model_dir: /models
auto_discover: true

# KV cache defaults
type_k: turbo3
type_v: turbo3
flash_attn: true
n_ctx: 8192

# VRAM
reserve_mb: 512

# Audit
audit_enabled: false
audit_path: /var/log/eie/audit.chain

log_level: info

# Model groups (aliases = GGUF filenames without extension, or `models:` map entries)
groups:
  - name: core
    models: [model-a, model-b, model-c]
    required_responses: 3
    type: parallel
    pinned: true
    fallback: partial
```

The parser is a minimal YAML subset: flat `key: value` entries, a `groups:`
list, and a `models:` map (alias → path). See `presets/` for ready-to-use
configurations (`dual-core-six.yaml` is a **skeleton**: strategy only, no groups or
models defined yet).

## Tested Configurations

| OS | GPU | CUDA/ROCm | Models | VRAM | Status |
|---|---|---|---|---|---|
| Windows 11 | RTX 4090 Laptop 16 GB | CUDA 13.2 | Gemma 4 E2B + E4B | 7.5 GB | ✅ |
| Windows 11 | RTX 4090 Laptop 16 GB | CUDA 13.2 | Gemma 4 E4B solo | 5.6 GB | ✅ |
| Windows 11 | RTX 4090 Laptop 16 GB | CUDA | Gemma 4 26B A4B QAT Q4_0, 16k context | 15.3-15.6 GB | ✅ |
| Ubuntu 24 | RTX 4090 24 GB | CUDA 12.x | Various | — | ✅ |
| Linux | AMD GPUs | ROCm 6.x | — | — | 🎯 Target |
| macOS 15 (Intel x86_64) | CPU (Metal off) | N/A | Gemma 4 E2B QAT Q4_0, native chat template, Android-emulator client | RAM | ✅ |
| Android 15 | Snapdragon 8 Elite — Hexagon HTP v79 | N/A | Gemma 4 E2B QAT Q4_0 | ~2.5 GB RAM | ✅ |
| Android 14 | Snapdragon 8 Gen 3 — HTP v75 / Adreno 750 / CPU i8mm | N/A | Gemma 4 E2B QAT Q4_0 + Q4_K_M | ~3 GB RAM | ✅ |
| Android 13/15 | Snapdragon 888, Dimensity 9000+ — CPU | N/A | Gemma 4 E2B, Ministral 3 3B | RAM | ✅ |
| Any | CPU only | N/A | Any GGUF | RAM | ✅ |

## VRAM Budget Estimates

> **Not measured.** Only two-model co-residency has been measured to date (Gemma 4 E2B + E4B, ~7.5 GB — see Performance). The rows below are projections from per-model footprints, not benchmark results.

With TurboQuant turbo3 (Q4_K_M weights, 4096 context):

| Scenario | GPU | Models | VRAM | Margin |
| --- | --- | --- | --- | --- |
| 3-model group | RTX 4090 16 GB | 7B + 3B + 2.4B | ~7.7 GB | ~8.3 GB |
| 6-model dual-core | AMD W7900 48 GB | 2×3 LLMs | ~16 GB | ~32 GB |
| 6 LLMs + vision | AMD W7900 48 GB | 6 + vision 2B | ~18 GB | ~30 GB |
| Fixed appliance | Any 8-16 GB | 2-4 models | ~5-8 GB | ~3-8 GB |

## Migration from Ollama

1. Build EIE for your GPU
2. Download GGUF models from HuggingFace (e.g., `bartowski/google_gemma-4-E4B-it-GGUF`)
3. Start EIE: `./build/eie-server -m model.gguf --port 8090`
4. Point your application to `http://localhost:8090/v1` instead of `http://localhost:11434/v1`
5. Same API, faster inference, lower VRAM

> **Note:** Ollama stores models as split blobs which are not directly compatible with EIE. Download GGUF files directly from HuggingFace instead.

## Docker

```bash
# NVIDIA
docker compose -f docker/docker-compose.yaml up -d eie-cuda

# AMD
docker compose -f docker/docker-compose.yaml up -d eie-rocm
```

## Project Structure

```
eie/
├── backends/           # Compute backend abstraction
├── core/               # Engine core (scheduling, model mgr, VRAM)
├── server/             # API server (OpenAI Layer 1 + Extensions)
├── monitoring/         # Health, metrics, audit
├── presets/            # Ready-to-use YAML configs
├── scripts/            # Build scripts (Linux + Windows)
├── contrib/            # Community extensions
├── docker/             # Dockerfiles (CUDA + ROCm)
├── mobile/             # EIE Mobile: embedded Android engine (JNI wrapper + arm64 build recipes)
├── tests/              # API tests
├── llama.cpp/          # Git submodule
├── CMakeLists.txt
├── LICENSE             # Apache 2.0
└── NOTICE              # Attributions
```

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md). Custom scheduling strategies and GPU backends are welcome in `contrib/`.

All contributions must be Apache 2.0 compatible.

## Acknowledgments

* [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov — inference engine foundation
* [TurboQuant](https://github.com/TheTom/turboquant_plus) by TheTom — KV cache compression
* [TurboQuant paper](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Google Research, ICLR 2026

## Citing EIE

If you use EIE in your research, please cite:

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

## License

```
Apache License 2.0
Copyright 2026 Elyne Corp
```
