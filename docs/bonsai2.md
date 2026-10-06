# Bonsai 2 with EIE (optional PrismML runtime)

This source-build profile runs **text-only** inference for Bonsai 2 GGUF models through EIE's HTTP API. It selects the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) at release [`prism-b10743-adfffbe`](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10743-adfffbe), commit `adfffbe41b2cabcd51fff326ab045662265062bb`. The normal EIE build continues to use its TurboQuant submodule. Model weights are downloaded separately and are not included in EIE.

Bonsai 2 needs the fork's Hadamard and sign-flip implementation. Stock llama.cpp rejects the `PQ2_0` and `PTQ1_0` files. A separate development `Q2_0` file can load in stock llama.cpp but produce incorrect output; successful loading is not a compatibility check. See PrismML's [backend and format support matrix](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/BACKEND-SUPPORT.md) and [format guide](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/MODEL-FORMATS.md).

## Get the source and weights

From a fresh checkout:

```bash
git clone https://github.com/deharoalexandre-cyber/EIE.git
cd EIE
git submodule update --init llama-prism
git -C llama-prism rev-parse HEAD
```

The last command should print `adfffbe41b2cabcd51fff326ab045662265062bb`. This profile does **not** use `patches/ews-runtime-2168b0.patch`.

Download one GGUF from [PrismML's Bonsai 2 model repository](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf). For example, with the [Hugging Face `hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli):

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir models/bonsai2
```

`PQ2_0` is the PrismML demo's default and is about 7.2 GB. `PTQ1_0` is about 6.0 GB and can be downloaded by substituting `Ternary-Bonsai-2-27B-PTQ1_0.gguf`. The GGUF repository also contains a vision projector; EIE's current request parser does not accept image inputs, so it is not needed for this profile. Budget memory for the model, KV cache, runtime and any GPU offload in addition to the file size. Start with an 8,192-token context and increase it only after measuring memory use.

PrismML reports a `PQ2_0` load crash on some AVX-512 CPUs in this pinned release; the fix is on the newer `prism` source branch. If affected, use `PTQ1_0` with this pinned profile or update the fork only after rebuilding and validating EIE against the newer revision. See [known issues](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/main/KNOWN_ISSUES.md#cpu-crash-on-load-avx-512-cpus).

## Build EIE

Run one of these recipes from the EIE root. CMake and a C++17 compiler are required. GPU builds also need the corresponding SDK or toolkit. `EIE_LLAMA_RUNTIME=prism` selects the pinned `llama-prism` submodule; no PrismML prebuilt binary is linked into EIE.

### Linux

```bash
# CPU
cmake -S . -B build-prism -DEIE_LLAMA_RUNTIME=prism -DLLAMA_OPENSSL=OFF
cmake --build build-prism --target eie-server -j 6

# NVIDIA CUDA instead: add -DGGML_CUDA=ON to the configure command.
# AMD ROCm/HIP instead: add -DGGML_HIP=ON to the configure command.
```

### macOS

```bash
# Apple Silicon / Metal
cmake -S . -B build-prism -DEIE_LLAMA_RUNTIME=prism -DGGML_METAL=ON -DLLAMA_OPENSSL=OFF
cmake --build build-prism --target eie-server -j 6

# Intel CPU instead: configure with -DGGML_METAL=OFF.
```

### Windows (Developer PowerShell for Visual Studio 2022)

```powershell
# CPU
cmake -S . -B build-prism -G "Visual Studio 17 2022" -A x64 -DEIE_LLAMA_RUNTIME=prism -DGGML_CUDA=OFF -DLLAMA_OPENSSL=OFF
cmake --build build-prism --config Release --target eie-server -j 6

# NVIDIA CUDA instead: configure with -DGGML_CUDA=ON and a supported CUDA toolkit.
```

These are build recipes for the optional integration, not performance or device validation results. The [PrismML support matrix](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/BACKEND-SUPPORT.md) records quantized-format kernels by backend for its pinned release.

## Run a text request

The included [`presets/bonsai2.yaml`](../presets/bonsai2.yaml) binds to loopback, selects F16 KV and starts with an 8,192-token context. F16 KV is explicit because EIE's standard default is `turbo3`, which belongs to its other llama.cpp fork. Start the model with the file path you downloaded; raise `--ctx` only after measuring memory use:

```bash
# Linux / macOS
./build-prism/eie-server --config presets/bonsai2.yaml \
  -m models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf
```

```powershell
# Windows (shared-library build)
$env:PATH = "$(Resolve-Path .\build-prism\bin\Release);$env:PATH"
.\build-prism\Release\eie-server.exe --config .\presets\bonsai2.yaml `
  -m .\models\bonsai2\Ternary-Bonsai-2-27B-PQ2_0.gguf
```

The CLI registers the model under the GGUF filename without `.gguf`. Check `GET /v1/models` after loading, then send a text request:

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Ternary-Bonsai-2-27B-PQ2_0","messages":[{"role":"user","content":"Hello"}],"enable_thinking":true,"reasoning_effort":"medium","temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.05,"repetition_penalty":1.0,"presence_penalty":0.0,"max_tokens":4096}'
```

PrismML's [demo launcher](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/scripts/start_llama_server.sh) recommends temperature `1.0`, top-p `0.95`, top-k `20`, min-p `0.05`, repetition penalty `1.0` and presence penalty `0.0` for thinking mode. EIE keeps its earlier defaults for other clients, so pass these values explicitly for Bonsai 2. `enable_thinking` and `reasoning_effort` are optional EIE request fields; the model's raw thinking text appears in `message.content` or SSE deltas, not a separate `reasoning_content` field.

The 8,192-context / 4,096-output example is a short smoke test and may stop during reasoning before the final answer. For longer reasoning, [PrismML recommends](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/main/KNOWN_ISSUES.md#empty-cut-off-or-runaway-output) `--ctx 65536` with `max_tokens` at least `16384`, subject to available memory. Use `reasoning_effort: "medium"`; PrismML reports `"high"` is unsupported for this model. Vision and native tool-call round trips are not exposed by EIE. Compare outputs with the PrismML demo before relying on model-specific behavior.

## Local validation (7 October 2026)

On Windows 11 with an i9-14900HX, MSVC 19.44 and a CPU-only PrismML build, `eie-server` loaded the 5,946,648,928-byte `PTQ1_0` GGUF (SHA-256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`) at `--ctx 2048`. `GET /v1/models` listed it. A non-thinking HTTP request for `42` returned `42`; a second request with `enable_thinking: true`, `reasoning_effort: "medium"` and the sampling parameters above returned reasoning text followed by `42`. The three offline serving tests and `serving-model-contract --api-only` passed. This verifies a bounded text route on this host; no PQ2_0, GPU, mobile, vision, throughput or long-context qualification is claimed.

EIE's source is [Apache 2.0](../LICENSE), the PrismML llama.cpp fork is [MIT](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10743-adfffbe/LICENSE), and the Bonsai 2 model repository declares [Apache 2.0](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf). Each artifact retains its own license.
