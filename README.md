# EIE — Elyne Inference Engine

**A generic, policy-driven, multi-model GGUF inference server.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-brightgreen.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ROCm](https://img.shields.io/badge/AMD-ROCm-ED1C24.svg)](https://www.amd.com/en/products/software/rocm.html)
[![Windows](https://img.shields.io/badge/Windows-CUDA%2013.2-0078D6.svg)](#windows-cuda)

---

EIE is a local inference server that loads GGUF models, serves them via an OpenAI-compatible REST API, and manages GPU memory. It is designed as **infrastructure** — it serves completions, nothing more. Orchestrators, agents, and domain-specific logic are clients of this server.

## Performance

Benchmarked on NVIDIA GeForce RTX 4090 Laptop GPU (16 GB VRAM), Windows 11:

| Model | Quant | VRAM | Prompt eval | Generation |
|---|---|---|---|---|
| Gemma 4 E2B | Q6_K | ~2.5 GB | 3,146 t/s | 126 t/s |
| Gemma 4 E4B | Q6_K | ~4.5 GB | 1,883 t/s | 70 t/s |
| **Both loaded** | Q6_K | **~7.5 GB** | — | — |

Compared to Ollama with the same models on the same hardware:
- **30% less VRAM** (7.5 GB vs 10.8 GB)
- **2x faster generation** (126 t/s vs ~60 t/s on E2B)
- Prompt cache active: `sim_best = 0.876` — subsequent requests are faster

## Why EIE?

|  | Ollama | vLLM | llama.cpp server | **EIE** |
| --- | --- | --- | --- | --- |
| **Scheduling** | None (FIFO) | Continuous batching | None (FIFO) | Policy-driven (pluggable) |
| **Model Groups** | No | No | No | Parallel, Sequential, Fan-out |
| **Fallback** | No | No | No | strict / partial / retry / replace |
| **KV Cache** | f16 / q8 / q4 | f16 / FP8 | f16 / q8 / q4 + TurboQuant | All legacy + **TurboQuant turbo2/3/4** |
| **Adaptive KV** | No | No | No | Health-check → auto downgrade |
| **Multi-model** | Sequential (swap) | Single model | Single model | **Simultaneous under constraints** |
| **Windows** | Yes | No | Yes | **Yes (CUDA 13.2 validated)** |
| **NVIDIA** | CUDA | CUDA | CUDA | CUDA (native) |
| **AMD** | Experimental | Partial | Partial | **ROCm first-class** |
| **VRAM mgmt** | Opaque | Per-request | None | Per-group budgets + watermarks |
| **Audit** | No | No | No | Hash-chained audit trail |
| **License** | MIT | Apache 2.0 | MIT | **Apache 2.0** |

## Key Features

### Policy Engine (pluggable)

Scheduling behavior is defined by strategies, not hardcoded. Four built-in strategies ship with EIE. Custom strategies can be loaded from shared libraries without recompiling.

```yaml
policy:
  strategy: pinned-group        # or: generic, multi-group, fixed-appliance
  # strategy: plugin:libcustom.so  # custom plugin
```

| Strategy | Behavior | Use case |
| --- | --- | --- |
| `generic` | On-demand loading, LRU eviction, FIFO | Ollama replacement |
| `pinned-group` | N models pinned, multi-response required | Multi-model deliberation |
| `multi-group` | Multiple pinned groups, each with own rules | Dual-core architectures |
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
**Fan-out** — Same prompt to N models, best response selected.

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

```yaml
vram:
  reserve_mb: 512           # always keep free
  low_watermark: 85         # start evicting non-pinned
  critical_watermark: 95    # force eviction
  group_isolation: true     # per-group VRAM budgets
```

### Compute Backend Abstraction

One codebase. CUDA, ROCm, and CPU detected automatically at runtime.

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

# Configure and build
cmake -B build -G "Visual Studio 17 2022" -DGGML_CUDA=ON -DCUDAToolkit_ROOT="$env:CUDA_PATH" -DCMAKE_CUDA_COMPILER="$env:CUDA_PATH\bin\nvcc.exe"
cmake --build build --config Release
```

> **Note:** Build takes 15-20 minutes due to CUDA kernel compilation. The binary is at `build\bin\Release\llama-server.exe`.

## Quick Start

### Single model

```bash
# Linux
./build/eie-server -m model.gguf -c 8192 --port 8090 -ngl 99

# Windows
build\bin\Release\llama-server.exe -m model.gguf -c 8192 --port 8090 -ngl 99
```

### Multi-model router

Load all GGUF models from a directory and route by model name via the API:

```bash
# Linux
./build/eie-server --models-dir /path/to/models -c 8192 --port 8090 -ngl 99

# Windows
build\bin\Release\llama-server.exe --models-dir C:\Users\User\models -c 8192 --port 8090 -ngl 99
```

### Web UI

EIE includes a built-in web interface. Open your browser to `http://127.0.0.1:8090` after starting the server.

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

| Endpoint | Method | Description |
| --- | --- | --- |
| `/v1/chat/completions` | POST | Chat completion (streaming supported) |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List available models |
| `/v1/embeddings` | POST | Embeddings |
| `/health` | GET | Server health |

### Layer 2 — Generic Extensions

| Endpoint | Method | Description |
| --- | --- | --- |
| `/v1/batch/execute` | POST | Execute a model group (N parallel responses) |
| `/v1/chain/execute` | POST | Execute a sequential chain (pipeline) |
| `/v1/admin/models/load` | POST | Load a GGUF model into VRAM |
| `/v1/admin/models/unload` | POST | Unload a model |
| `/v1/admin/models/discover` | GET | Scan model directory |
| `/v1/admin/vram/status` | GET | VRAM per GPU, per model, per group |
| `/v1/admin/scheduling/status` | GET | Active policy and groups |
| `/v1/admin/config/reload` | POST | Hot-reload YAML configuration |
| `/metrics` | GET | Prometheus-compatible metrics |

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
```

See `presets/` for ready-to-use configurations.

## Tested Configurations

| OS | GPU | CUDA/ROCm | Models | VRAM | Status |
|---|---|---|---|---|---|
| Windows 11 | RTX 4090 Laptop 16 GB | CUDA 13.2 | Gemma 4 E2B + E4B | 7.5 GB | ✅ |
| Windows 11 | RTX 4090 Laptop 16 GB | CUDA 13.2 | Gemma 4 E4B solo | 5.6 GB | ✅ |
| Ubuntu 24 | RTX 4090 24 GB | CUDA 12.x | Various | — | ✅ |
| Linux | AMD GPUs | ROCm 6.x | — | — | 🎯 Target |
| Any | CPU only | N/A | Any GGUF | RAM | ✅ |

## VRAM Budget Examples

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
3. Start EIE: `./build/eie-server -m model.gguf --port 8090 -ngl 99`
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
