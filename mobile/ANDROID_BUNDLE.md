# EIE Android arm64 CPU bundle

This is the native EIE HTTP engine and its shared libraries, not an APK or a copy
of Elyne Mobile. No model weights, API keys, memory or conversations are included.
The existing JNI wrapper is a separate integration path and is not compiled in
this bundle. This package does not qualify Android EWS, OpenCL, Hexagon or vision.

## Select the CPU variant

- `dotprod`: ARMv8.2-A with dot-product and FP16 vector instructions.
- `i8mm`: ARMv8.6-A with dot-product and integer matrix multiplication, no SVE.

Both use NEON/ASIMD. Neither is a universal binary for every ARM64 Android device.
The compile target is Android API 26 or newer; see the device receipt for actual
tested OS/SoC combinations. GPU/NPU offload and OpenMP are disabled in this profile.
All inference libraries and their hashes are identified in `manifest.json`.

## Run on a device

Extract one bundle, push its `bin` and `lib` directories to a dedicated directory
under `/data/local/tmp/`, and provide your own compatible GGUF model. For example,
from an ADB shell in that directory, with your model at `model.gguf`:

```sh
chmod 755 bin/eie-server
LD_LIBRARY_PATH="$PWD/lib" ./bin/eie-server --config eie.yaml
```

Example `eie.yaml` (replace the model path with its absolute device path):

```yaml
host: 127.0.0.1
port: 18761
auto_discover: false
type_k: f16
type_v: f16
flash_attn: true
n_ctx: 1024
models:
  resident: /data/local/tmp/eie-demo/model.gguf
threads:
  resident: 4
preload: [resident]
```

Forward the port with `adb forward tcp:18761 tcp:18761`, then call the engine:

```sh
curl http://127.0.0.1:18761/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"resident","messages":[{"role":"user","content":"Reply with 42."}],"temperature":0,"max_tokens":32,"stream":true}'
```

This uses the same limited OpenAI-shaped API as desktop EIE. Only the specific
chat cases in the device receipt are qualified, not every API feature.

## Rebuild and retain raw evidence

From an EIE source checkout, initialize the pinned llama.cpp submodule and apply
`patches/ews-runtime-2168b0.patch` once. Then use Python 3.11+, CMake, Ninja and an
Android NDK:

```sh
python scripts/build_android_neon.py --ndk /path/to/android-ndk --variant i8mm
python scripts/build_android_neon.py --ndk /path/to/android-ndk --variant dotprod
```

To retain an earlier package when rebuilding, pass `--output-dir` with a new
destination. The build tree is reused, but package files are never overwritten.

The builder writes compile logs, ELF dependency metadata, licenses, headers and
SHA-256 inventories beside the native server. A release device receipt is separate
from a successful build. `validate_android_neon.py --help` describes on-device
validation against a model already accessible to the ADB shell. It creates an
isolated test directory, records raw process/HTTP outputs and only stops its own
server. It does not install, stop or modify Elyne Mobile.

The `source_revision`, runtime pin, published runtime patch and build recipe hash
identify the source used. A different NDK/toolchain can change the output hashes.
`source_dirty` covers tracked EIE files; the intentionally patched submodule is
identified separately by its pin, patch hash and `runtime_diff_sha256`.
Model name, quantization and SHA-256 must be retained per run; do not pool timings
across QAT Q4_0 and Q4_K_M or infer a CPU/GPU/NPU ranking.

## Licenses

EIE is Apache-2.0; llama.cpp and its vendored dependencies retain their licenses.
The shipped C++ runtime retains its Android NDK/LLVM notices. See `licenses/`.
GGUF weights are supplied by the user under their own license.
