# EIE — Elyne Inference Engine

**A generic, policy-driven, multi-model GGUF inference server.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-brightgreen.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ROCm](https://img.shields.io/badge/AMD-ROCm-ED1C24.svg)](https://www.amd.com/en/products/software/rocm.html)

---

EIE is a local inference server that loads GGUF models, serves them via an OpenAI-compatible REST API, and manages GPU memory. It is designed as **infrastructure** — it serves completions, nothing more. Orchestrators, agents, and domain-specific logic are clients of this server.

## Why EIE?

|  | Ollama | vLLM | llama.cpp server | **EIE** |
|---|---|---|---|---|
| **Scheduling** | None (FIFO) | Continuous batching | None (FIFO) | Policy-driven (pluggable) |
| **Model Groups** | No | No | No | Parallel, Sequential, Fan-out |
| **Fallback** | No | No | No | strict / partial / retry / replace |
| **KV Cache** | f16 / q8 / q4 | f16 / FP8 | f16 / q8 / q4 + TurboQuant | All legacy + **TurboQuant turbo2/3/4** |
| **Adaptive KV** | No | No | No | Health-check → auto downgrade |
| **Multi-model** | Sequential (swap) | Single model | Single model | **Simultaneous under constraints** |
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
|---|---|---|
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
|---|---|---|---|
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

```bash
# NVIDIA
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

# AMD
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release

# CPU (fallback)
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

### Audit Trail (optional)

Hash-chained inference log for compliance-sensitive deployments. Each group execution produces a tamper-evident record.

```yaml
audit:
  enabled: true
  path: /var/log/eie/audit.chain
```

## Quick Start

```bash
# Clone
git clone https://github.com/[org]/eie.git
cd eie && git submodule update --init

# Build (CUDA)
./scripts/build-cuda.sh

# Place your GGUF models in /models (or configure model_dir)
mkdir -p /models
cp your-model.Q4_K_M.gguf /models/

# Run
./build/eie-server --config presets/generic.yaml
```

## API

### Layer 1 — OpenAI Compatible (drop-in)

Any OpenAI-compatible client works without modification.

```bash
# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completion (streaming supported) |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List available models |
| `/v1/embeddings` | POST | Embeddings |
| `/health` | GET | Server health |

### Layer 2 — Generic Extensions

| Endpoint | Method | Description |
|---|---|---|
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
curl http://localhost:8080/v1/batch/execute \
  -H "Content-Type: application/json" \
  -d '{
    "group": "core",
    "messages": [{"role": "user", "content": "Analyze this alert"}]
  }'

# Response
{
  "group": "core",
  "responses": [
    {"model": "model-a", "content": "...", "latency_ms": 1200},
    {"model": "model-b", "content": "...", "latency_ms": 980},
    {"model": "model-c", "content": "...", "latency_ms": 1450}
  ],
  "completed": 3,
  "required": 3,
  "status": "complete"
}
```

## Configuration

```yaml
# /etc/eie/engine.yaml
host: 0.0.0.0
port: 8080
strategy: pinned-group
model_dir: /models
auto_discover: true

# KV cache defaults
type_k: turbo3
type_v: turbo3
flash_attn: true
n_ctx: 4096

# VRAM
reserve_mb: 512

# Audit
audit_enabled: false
audit_path: /var/log/eie/audit.chain

log_level: info
```

See `presets/` for ready-to-use configurations:
- `generic.yaml` — Ollama replacement
- `three-model-group.yaml` — Pinned 3-model deliberation
- `dual-core-six.yaml` — 2×3 model dual-core
- `fixed-appliance.yaml` — Edge device, no dynamic loading
- `development.yaml` — Local development (CPU, f16, debug)

## Docker

```bash
# NVIDIA
docker compose -f docker/docker-compose.yaml up -d eie-cuda

# AMD
docker compose -f docker/docker-compose.yaml up -d eie-rocm
```

## VRAM Budget Examples

With TurboQuant turbo3 (Q4_K_M weights, 4096 context):

| Scenario | GPU | Models | VRAM | Margin |
|---|---|---|---|---|
| 3-model group | RTX 4090 16 GB | 7B + 3B + 2.4B | ~7.7 GB | ~8.3 GB |
| 6-model dual-core | AMD W7900 48 GB | 2×3 LLMs | ~16 GB | ~32 GB |
| 6 LLMs + vision | AMD W7900 48 GB | 6 + vision 2B | ~18 GB | ~30 GB |
| Fixed appliance | Any 8-16 GB | 2-4 models | ~5-8 GB | ~3-8 GB |

## Project Structure

```
eie/
├── backends/           # Compute backend abstraction
│   ├── compute_backend.h   # Abstract interface
│   └── cpu_backend.cpp     # CPU + CUDA + HIP implementations
├── core/               # Engine core
│   ├── scheduling.h        # PolicyStrategy + GroupScheduler
│   ├── model_manager.h     # Registry, load/unload, discovery
│   ├── vram_manager.h      # Budgets, watermarks, QoS
│   └── config.h            # Configuration parser
├── server/             # API server
│   ├── main.cpp            # Entry point
│   ├── api.h / api.cpp     # OpenAI Layer 1 + Extensions Layer 2
├── monitoring/         # Health, metrics, audit
│   └── monitoring.h        # Prometheus + hash-chain audit
├── presets/             # Ready-to-use YAML configs
├── contrib/             # Community extensions
│   ├── strategies/         # Custom scheduling plugins
│   └── bindings/           # Python, Rust, Go wrappers
├── scripts/             # Build scripts
├── docker/              # Dockerfiles (CUDA + ROCm)
├── tests/               # API tests
├── llama.cpp/           # Git submodule (TurboQuant fork)
├── CMakeLists.txt
├── LICENSE              # Apache 2.0
└── NOTICE               # Attributions
```

## Migration from Ollama

1. Build EIE for your GPU
2. Copy GGUF models from `~/.ollama/models/` to `/models/`
3. Use `presets/generic.yaml` (or write your own)
4. Start EIE on port 8080
5. Point your clients to `http://localhost:8080/v1/chat/completions`
6. Enable TurboQuant: set `type_k: turbo3`, `type_v: turbo3`
7. Enjoy 5× KV cache compression and multi-model groups

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md). Custom scheduling strategies and GPU backends are welcome in `contrib/`.

All contributions must be Apache 2.0 compatible.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov — inference engine foundation
- [TurboQuant](https://github.com/TheTom/turboquant_plus) by TheTom — KV cache compression
- [TurboQuant paper](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Google Research, ICLR 2026

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
