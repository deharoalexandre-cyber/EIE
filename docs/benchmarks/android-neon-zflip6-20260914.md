# Android CPU bundles: Z Flip6, Gemma E2B QAT

**14 September 2026. Locally verified by the maintainers, not independent replication.**

Both released native EIE CPU variants (`dotprod` and `i8mm`) run text chat on
the Galaxy Z Flip6 with the same **Gemma 4 E2B QAT Q4_0** file. This is a working
HTTP engine bundle, not an APK or a repackaging of Elyne Mobile. No model weights,
application memory, conversations or personal device history are distributed.

- [Download both bundles and raw evidence](https://github.com/deharoalexandre-cyber/EIE/releases/tag/android-neon-2026.09.14)
- [Run and rebuild instructions](../../mobile/ANDROID_BUNDLE.md)
- [Every request, device/build/model identity and timings (JSON)](data/android-neon-zflip6-20260914.json)
- [Raw ADB/server/HTTP outputs and manifests (ZIP)](https://github.com/deharoalexandre-cyber/EIE/releases/download/android-neon-2026.09.14/android-neon-zflip6-20260914-raw.zip)

## Exact configuration

| Item | Value |
|---|---|
| Device | Samsung Galaxy Z Flip6, SM-F741B |
| SoC / platform | SM8650 (Snapdragon 8 Gen 3), pineapple |
| OS | Android 16, API 36, security patch 2026-08-05 |
| Memory | `/proc/meminfo`: 11,381,404 kB total, 12 GB-class device |
| Model | Gemma 4 E2B QAT Q4_0; **not Q4_K_M** |
| Model size | 3,349,514,112 bytes |
| Release source | `50f49079a6a5a3e1cc6f01048329ec608322c475` |
| Runtime | `2168b0cd8b87c75c29a1e6588692ebbb805b9bd2` + `patches/ews-runtime-2168b0.patch` |
| Build | Android NDK 28.2.13676358 (r28c), Clang 19.0.1, Release, arm64-v8a, API 26 target |
| CPU variants | `armv8.2-a+dotprod+fp16`; `armv8.6-a+dotprod+i8mm+nosve` |
| Execution | CPU only, 4 threads, OpenMP off, F16 K/V, context 1,024 |
| Requests | Temperature 0, `one_shot: true`, `truncate_prompt: false`, `strict_model: true` |

Both variants require the stated CPU instruction features; this is not a
universal ARM64 binary. API 26 is the compile target, not an Android 8 test.
Neither OpenCL, Hexagon, CUDA nor Metal is enabled. EWS is not exercised.

Model SHA-256 (identical in all four runs):

```text
3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd
```

The model was copied into an isolated ADB test directory and rehashed. Elyne
Mobile was not replaced or stopped; its process was still alive after testing.
Its background work and normal device activity were not controlled. Tests use
a dedicated loopback server/ADB forward and stop only their own server.

## What passed

For **each release variant**:

1. Deployed binary and all shared-library hashes match the bundle manifest.
2. Native `eie-serving-tests`: **628 assertions pass**. These are no-model tests;
   their placeholder output is not counted as inference evidence.
3. The real EIE process loads the hash-identified QAT model and exposes `resident`.
4. `What is 17 + 25? Reply with only the integer.` produces `42` with
   `finish_reason: stop`, identically buffered and streamed. The SSE terminator
   and returned usage are retained in raw responses.
5. Three subsequent streamed generations produce 64 sampled tokens each, with
   the explicit `length` finish reason. They use the same short bicycle prompt.
6. The test-owned process exits and its port forward is removed.

This is two prompts, not a general answer-quality suite. The bicycle request
asks for about 100 words but caps output at 64 tokens intentionally: these are
**incomplete responses used to exercise length termination**, not successful
100-word answers. The arithmetic output has three sampled tokens including the
terminal token, not three visible words.

## All recorded generation trials

The rate is **API completion tokens / whole host request wall time**. It includes
prefill and ADB/HTTP transport and is **not a decode-only rate**. TTFC measures
the first nonempty SSE content received by the host, not server-internal TTFT.

Each cell lists the three consecutive 64-token trials, in execution order:

| Run | Source | Request wall seconds | First content seconds | Tokens / request-wall-second |
|---|---|---|---|---|
| Preliminary i8mm | `8186c76` | 7.438, 4.750, 5.047 | 0.453, 0.469, 0.454 | 8.60, 13.47, 12.68 |
| Preliminary dotprod | `8186c76` | 4.844, 5.015, 5.922 | 0.562, 0.765, 0.594 | 13.21, 12.76, 10.81 |
| Release i8mm | `50f4907` | 8.125, 13.828, 9.219 | 0.625, 0.359, 0.500 | 7.88, 4.63, 6.94 |
| Release dotprod | `50f4907` | 6.422, 5.500, 5.609 | 0.610, 0.641, 0.687 | 9.97, 11.64, 11.41 |

**All four device runs pass the functional criteria.** The JSON also retains
all eight arithmetic requests, including the 4.093-second first preliminary
buffered request. No timing was dropped for being slow.

Do not pool these into a speedup: runs were sequential, not randomized, and
the release rebuild incorporates the intervening macOS-source changes. Cache,
charging, temperatures and background load were not held constant. `one_shot`
creates a fresh inference context; it does not flush the OS page cache. The
observed variability is real, but its cause is not isolated by this campaign.

## Binary identity and raw evidence

Release binary SHA-256:

| Variant | File | SHA-256 |
|---|---|---|
| i8mm | `bin/eie-server` | `e71142a4a6091ec3d17bb09cd1c95f46a9fef09ce412f4b4f1f639e2b14eafc6` |
| i8mm | `lib/libggml-cpu.so` | `8c962010c0bbf9a21f3483f559954b438d995ad72d44e24f72fc1f8fd760c3fa` |
| dotprod | `bin/eie-server` | `c51cff822309a8b077b5635a52599f6e4d921d222945751adb6c4196bea1e4dd` |
| dotprod | `lib/libggml-cpu.so` | `6fcda19a9fcfdc288eab3a834ba080be4d80c420097b7d3d8f4b464011c5a33f` |

Every shared library, NDK C++ runtime, header, notice, build log and ELF report
has a hash in the bundle manifest/inventory. The JSON repeats device, variant,
model quantization and binary/library identity **for each request**, so rates
cannot silently lose their configuration. Preliminary binaries remain identified
by their own manifests rather than being relabelled as the final release.

The raw evidence archive contains byte-identical dedicated ADB stdout/stderr,
server output, exact requests, HTTP/SSE replies, deployment hashes, device
properties, memory/thermal snapshots and build evidence. `inventory.json`
provides SHA-256 and byte length for each contained file. The public reports
index those raw files; they do not replace them with summaries.

Privacy exclusion: the initial validator's broad Samsung battery dump also
contained unrelated charging history. Those four battery files per preliminary
run are omitted and explicitly listed as exclusions; inference outputs are
unchanged. The release validator requests only current battery level,
temperature and USB-power state. Device serials and app conversations are not
part of the published evidence.

Pre-device build attempts exposed shell argument quoting and an incomplete
license-packaging step. Both were corrected before the recorded device runs;
the incomplete package is not a release asset. The preliminary successful runs
remain available instead of being replaced by the final, sometimes slower runs.

Release archive checksums:

```text
36851959151636b77050cab22c0980f0097cbaa36bf458b969266c5bb612c520  eie-android-arm64-i8mm.zip
996b873aa7dfd6c200d19fc48509fe1c46e368b6ea77693a1218049e25dafb09  eie-android-arm64-dotprod.zip
ce26158dbe8751ffcc289b92e2f0ca9a60eaed89c0c2be8cb04c5d23b0f28d4a  android-neon-zflip6-20260914-raw.zip
```

## Replay on another authorized device

Extract the appropriate bundle and place your licensed GGUF in a shell-readable
device directory. From the bundle directory, with Python 3.11+ and ADB:

```sh
python validate_android_neon.py --adb adb --serial YOUR_DEVICE \
  --bundle . --model /data/local/tmp/eie-demo/model.gguf \
  --model-name "Gemma 4 E2B QAT" --quantization QAT_Q4_0 --out new-device-run
```

An existing output directory is not overwritten. The validator deploys its own
binary and libraries, checks their hashes, retains raw output and stops its own
server. A different model can produce a different arithmetic answer; record that
failure rather than changing the expected answer after observing it.

## Not established by this release

- Android EWS, large MoE support, GPU/NPU, quantized KV, vision or embeddings.
- JNI client/APK integration, application cognition or memory continuity.
- Long-context quality, cancellation/recovery, concurrent callers or fleet reliability.
- An i8mm-vs-dotprod, CPU-vs-NPU or CPU-vs-GPU performance advantage.
- The older six-row mobile timing table: its heterogeneous QAT/Q4_K_M builds
  remain maintainer-reported, not retroactively validated by this QAT CPU run.

The next useful step is broader device/workload testing with controlled cache,
thermal and concurrency conditions, not inferring general guarantees from one phone.
