# EIE Mobile: Android engine bundles and embedded components

The Android arm64 flavor of EIE uses a TurboQuant llama.cpp fork and a JNI
wrapper. It loads GGUF models and emits tokens; memory, identity, orchestration
and application behavior belong to the client.

**Publication status, 14 September 2026:** [native CPU bundles](https://github.com/deharoalexandre-cyber/EIE/releases/tag/android-neon-2026.09.14)
are available for arm64 `dotprod` and `i8mm`: EIE HTTP server, shared libraries,
headers, notices, manifests and a device-validation script. Both passed on a
Galaxy Z Flip6 / SM8650 / Android 16 with **Gemma 4 E2B QAT Q4_0**, not Q4_K_M.
[Bundle instructions](ANDROID_BUNDLE.md) / [raw-evidence receipt](../docs/benchmarks/android-neon-zflip6-20260914.md).

The native bundle is not an APK and does not replace the JNI integration below.
The older six-row performance table remains maintainer-reported; this new CPU
campaign does not retroactively qualify those devices, builds or backends.

## Contents and prerequisites

| Path | Purpose |
|---|---|
| `../scripts/build_android_neon.py` | Package the native EIE server and libraries with an explicit CPU instruction set |
| `../scripts/validate_android_neon.py` | Raw, isolated ADB and HTTP tests against a hash-identified GGUF |
| `jni/native_inference.cpp` | JNI wrapper and package forwarders |
| `jni/CMakeLists.txt` | Links externally supplied prebuilt fork libraries |
| `scripts/build_cpu_variants.sh` | Android CPU-variant build recipe |
| `scripts/build_hexagon_skels.sh` | Hexagon backend / HTP skel build recipe |

The recipes refer to external build paths and the
`ghcr.io/snapdragon-toolchain/arm64-android` image (NDK, OpenCL, Hexagon SDK).
The JNI include paths, libraries, application/Kotlin code and app manifest
must be supplied/adapted. An image name is not a frozen image digest.
The desktop EWS patch/campaign is not Android EWS qualification.

## Implemented wrapper paths

- **KV prefix reuse:** compares tokens with retained state and avoids
  re-evaluating an identical prefix. Changing earlier history invalidates reuse.
  This does not make tokenization, attention or the whole request O(new tokens).
- **CPU backend discovery:** loads libraries from `nativeLibraryDir` when
  normal backend discovery registers no devices. The supplied build recipe
  targets multiple CPU variants; actual coverage depends on packaged libraries.
- **OpenCL / Adreno:** when offloading, the wrapper selects F16 KV and
  flash-attention AUTO. Maintainers report incompatibility of quantized KV or
  forced flash-attention in the tested OpenCL profile; this is not a universal
  claim about every upstream version. Required entry points include
  `clCreateBufferWithProperties`.
- **Hexagon HTP:** device selection and FastRPC/skel integration paths exist.
  The app needs the vendor native-library declaration, including
  `libcdsprpc.so`, and compatible DSP libraries. Combined-device scheduling
  was problematic in the reported profile; use the exposed device filter.
- **One-shot generation:** attempts a dedicated secondary context. Allocation
  failure falls back to the primary context; concurrent use is not proven
  race-free and may affect retained chat state.
- **System-property tuning:** `debug.elyne.model`, `mmproj`, `ngl`,
  `ndev`, `dev`, `temp`, `topp`, `topk` under the same
  `debug.elyne.` prefix. Inspect the wrapper and restart the client after
  changing the intended profile.

CPU/NEON quantized-cache selection also depends on client-side code not
bundled here. Do not infer that every desktop KV mode is available on mobile.

## Historical reported performance: separate from the native bundle

Gemma 4 E2B, QAT Q4_0 unless noted. These are maintainer-reported field figures,
not a controlled cross-backend comparison or independently replicated results.

| SoC | Backend | Decode tok/s | Prefill tok/s |
|---|---|---:|---:|
| Snapdragon 8 Elite | HTP v79 | 19.6-20.6 | 860-959 |
| Snapdragon 8 Gen 3 | HTP v75 | 13.4-17.8 | 600-664 |
| Snapdragon 8 Gen 3 | CPU i8mm, Q4_K_M | 17.4-18.7 | 43-53 |
| Snapdragon 8 Gen 3 | Adreno 750, Q4_0 | 8.0-9.7 | 204-229 |
| Snapdragon 888 | CPU dotprod | 7.0-8.7 | ~31 |
| Dimensity 9000+ | CPU i8mm | ~6 | ~27 |

Bandwidth and compute bottlenecks are possible explanations, not causal
measurements in this table. Different quantizations, absent sample distributions
and incomplete device/build details preclude a general NPU/GPU/CPU ranking.

## Known limitations and next qualification

- Maintainer-reported HTP v68 behavior: session opens but q6_K tensors fail
  repacking; a stalled DSP queue can block outside an abortable loop.
- Older OpenCL drivers may lack required entry points. The reported device
  matrix does not guarantee operation on every ROM or SoC revision.
- No general concurrent-caller / secondary-context allocation-failure gate
  is provided. Do not describe one-shot isolation as unconditional.
- Reproducing the historical JNI/HTP/OpenCL rows still needs pinned build/model
  hashes, a minimal client and raw runs. The separate CPU HTTP bundles now include
  these build/model identities and raw test outputs for the Z Flip6 only; they do
  not establish decode-only rates or controlled thermal/cache repeatability.

See the [repository audit](../docs/CLAIMS_AUDIT.md). Hardware availability
and a build recipe alone do not certify production operation.
