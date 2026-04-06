# EIE — Elyne Inference Engine

Generic, policy-driven, multi-model GGUF inference server.

- **TurboQuant-native** KV cache compression (turbo2/3/4 + legacy)
- **CUDA + ROCm first-class** — single codebase, auto-detection
- **Model Groups** — parallel, sequential chains, fan-out execution
- **Policy Engine** — pluggable scheduling strategies
- **VRAM QoS** — per-group budgets, watermarks, reserve pool
- **Audit Trail** — optional hash-chained inference log
- **OpenAI-compatible API** — drop-in alternative to Ollama

## Build

```bash
git submodule update --init
./scripts/build-cuda.sh   # or build-rocm.sh / build-cpu.sh
./build/eie-server --config presets/generic.yaml
```

## License

Apache License 2.0
