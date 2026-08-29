# EIE Mobile — embedded Android engine

The on-device flavor of EIE: the same TurboQuant llama.cpp fork, compiled for `arm64-v8a`
and embedded in an Android app as a set of shared libraries driven by a thin JNI wrapper.
Same philosophy as the server: **this is infrastructure** — it loads GGUF models and serves
tokens. Orchestration, identity and product logic live in the client app, not here.

## Contents

| Path | What it is |
|---|---|
| `jni/native_inference.cpp` | The complete JNI wrapper (single `.so`, dual-package forwarders) |
| `jni/CMakeLists.txt` | Wrapper build — links the prebuilt fork libraries as IMPORTED |
| `scripts/build_cpu_variants.sh` | Docker recipe: multi-variant CPU backends (runtime dispatch) |
| `scripts/build_hexagon_skels.sh` | Docker recipe: Hexagon NPU backend + HTP DSP skels (v68/v75/v79) |

Both build scripts run inside the `ghcr.io/snapdragon-toolchain/arm64-android` image
(NDK + OpenCL SDK + Hexagon SDK) against the TurboQuant fork
([`llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant), same submodule
as the server).

## What the wrapper provides

- **Incremental generation with KV reuse.** The prompt is compared token-wise against the
  previously evaluated state; only the new suffix is decoded (`generate: reuse=N` logging).
  Contract with the client: the transcript must be **immutable** — anything already rendered
  into history is never modified or moved, and volatile context (timestamps, retrieved
  memory) is appended strictly at the tail. Cost drops from O(history) to O(new tokens).
- **Runtime CPU dispatch.** Built with `GGML_CPU_ALL_VARIANTS` + `GGML_BACKEND_DL`: one APK
  carries `libggml-cpu-android_armv8.x/v9.x` variants; the best one is selected at runtime.
  One binary covers ARMv8.2 (no i8mm — e.g. Snapdragon 888) through ARMv9 (i8mm/SVE2).
  Note: `ggml_backend_load_all()` scans the *executable* directory, which is empty on
  Android — the wrapper falls back to `ggml_backend_load_all_from_path(nativeLibraryDir)`
  when no device registered.
- **GPU (OpenCL / Adreno).** Full-graph offload works with two constraints discovered the
  hard way: quantized KV caches and forced flash-attention are not executable by the OpenCL
  backend (scheduler abort on pre-allocated tensors) — when offloading, the wrapper forces
  **KV f16 + flash-attn AUTO**. Requires OpenCL 3.0 drivers (`clCreateBufferWithProperties`).
- **NPU (Hexagon HTP).** FastRPC session against the bundled `libggml-htp-vXX.so` skels.
  Requires `<uses-native-library android:name="libcdsprpc.so" android:required="false"/>`
  in the app manifest (Android namespace isolation otherwise hides the vendor lib).
  With both Adreno and HTP registered, llama.cpp will **split the model across both** —
  catastrophic at batch 1 — so the wrapper exposes a device filter.
- **Live tuning via system properties** (no rebuild, `setprop` + app restart):
  `debug.elyne.model` / `debug.elyne.mmproj` (artifact override), `debug.elyne.ngl`,
  `debug.elyne.ndev` (Hexagon sessions), `debug.elyne.dev` (`htp`|`gpu` filter),
  `debug.elyne.temp/topp/topk` (sampling).

## Measured performance (Gemma 4 E2B, QAT Q4_0 unless noted)

| SoC | Backend | Decode (tok/s) | Prefill (tok/s) |
|---|---|---|---|
| Snapdragon 8 Elite | HTP v79 | **19.6–20.6** | **860–959** |
| Snapdragon 8 Gen 3 | HTP v75 | 13.4–17.8 | 600–664 |
| Snapdragon 8 Gen 3 | CPU i8mm (Q4_K_M) | 17.4–18.7 | 43–53 |
| Snapdragon 8 Gen 3 | Adreno 750 (Q4_0) | 8.0–9.7 | 204–229 |
| Snapdragon 888 | CPU dotprod variant | 7.0–8.7 | ~31 |
| Dimensity 9000+ | CPU i8mm | ~6 | ~27 |

Rule of thumb on phones: **decode is DRAM-bandwidth-bound** (every token re-reads the
weights; a strong CPU already saturates the bus), while **prefill is compute-bound**
(matrix accelerators win 4–20×). QAT Q4_0 checkpoints + HTP repack shrink the per-token
working set enough that the NPU also wins decode on recent SoCs.

## Known limitations

- Hexagon HTP: v68 (SD888) opens a session but rejects `q6_K` tensors at repack; a wedged
  DSP queue blocks the calling thread outside any abortable loop (recovery: kill the client
  process; the driver tears the session down). Watchdog/timeout hardening is on the list.
- OpenCL backend requires CL 3.0; older vendor drivers (e.g. Adreno 660 on some ROMs)
  expose only CL 2.x and are rejected cleanly at probe.
- One llama context is shared by all callers; concurrent one-shot generations clobber the
  chat KV state. A dedicated secondary context is the planned fix.
