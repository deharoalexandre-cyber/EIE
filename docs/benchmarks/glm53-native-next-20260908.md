# GLM 5.3 Flash laptop bring-up - 8 September 2026

Maintainer-side functional experiment. **This uses a separate native llama.cpp
runtime, not EIE or the Gemma EWS implementation.** No speed acceptance threshold.
The goal is a usable GLM answer and continuation by the same real Next 12B.

## Pinned artifacts and configuration

- Model: [Unsloth GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9/UD-Q4_K_XL),
  revision `621d456e93e926e4b52f85cff5f634358c1828f9`, UD-Q4_K_XL, six GGUF shards.
- Weight files: 199,707,321,347 bytes, approximately 199.7 decimal GB / 185.99 GiB.
  All six size/SHA-256 checks passed during download and an independent full reread.
- Native source: [Unsloth llama.cpp](https://github.com/unslothai/llama.cpp/tree/b9b8207fcfc2962093b9466df7af4ff29c2a81ef),
  revision `b9b8207fcfc2962093b9466df7af4ff29c2a81ef`, no source modifications.
  This is the GLM5next port associated with [upstream PR 27754](https://github.com/ggml-org/llama.cpp/pull/27754),
  not the runtime pinned by EIE.
- Hardware: i9-14900HX, 34,070,192,128 physical RAM bytes, RTX 4090 Laptop 16 GiB-class GPU.
- Build: Windows, MSVC 19.44, CUDA 13.2.78, SM89, Release shared libraries,
  `GGML_NATIVE=OFF`; NVIDIA driver 595.79.
- Model configuration: mmap, no repack, no fit adjustment, 20 GPU layers,
  all routed expert weights on CPU, 2048 context, batch 32 / microbatch 1,
  one slot, eight CPU threads. Flash attention, MTP and reasoning disabled.
  `NVIDIA_TF32_OVERRIDE=0` follows the native port's correctness guidance.

The server reports architecture `glm5next` and 320,759,404,382 total parameters.
The GGUF includes an unused MTP block. This is not an active-parameter measurement.

## Observed results

| Probe | Observation | Scope |
|---|---|---|
| Native loading | Server reports model loaded after 23.642 s | No cold-cache control |
| Native arithmetic | `17 + 25` returns `42`, clean SSE end / `stop` | 29 prompt tokens, 3 completion tokens including termination |
| Native latency | First content 65.875 s; full HTTP response 68.109 s | One small request, not a throughput guarantee |
| Next roundtrip, run 1 | Actual 12B tool call -> real GLM text -> 12B synthesis -> separate answer `55` to `21 + 34` | Functional integration succeeds; GLM response hits the test's 128-token ceiling |
| Run 1 auxiliary completion | `finish_reason=length`, `incomplete=true`, 446.516 s tool wall time | **Not a clean full-answer success**; truncated evidence retained |
| Run 1 coexistence | 12813 MiB sampled total GPU use, including desktop, with 12B + Nomic + GLM loaded | Not model-weight size or full-system memory usage |
| Next roundtrip, run 2 | Actual 12B tool call -> complete GLM analysis -> 12B synthesis -> separate answer `55` | **PASS**, actual tools, new empty state |
| Run 2 auxiliary completion | 188 tokens, `finish_reason=stop`, `incomplete=false`, 611.812 s tool wall time | Normal 768-token allowance; no latency threshold |
| Run 2 native timing | 190.890 s prefill / 86 uncached tokens; 419.121 s decode / 188 tokens | Approximately 0.45 tok/s; not a paired speed comparison with run 1 |
| Run 2 coexistence / cleanup | Maximum sampled total GPU usage 12826 MiB; resident continues; original state unchanged | Owned test processes stopped; GPU use returned to 584 MiB |

The second run passes both integration and clean auxiliary completion. The
resident independently formulates a slightly different auxiliary question in
each run; changing latency cannot be attributed solely to the token allowance.
The truncated first attempt is retained alongside the clean second result.

The [machine-readable receipt](data/glm53-native-next-20260908.json) includes exact
questions, GLM answers, resident syntheses, source/runtime/model hashes and timings.
Receipt SHA-256 (UTF-8 / LF): `6e037f146cf8176ed3795552e498017103c622156b3d827cd69fa2a5b1f2aeb3`.

## Actual Next envelope, not a fabricated model exchange

Next imports its real session/turn/tool implementation with a new empty state.
Its resident is Gemma 4 12B QAT Q4_0 with 16384-token configured context and Nomic
embedding. There are 23 exposed tools. No production memories or RAG documents
are copied, and production state hashes are unchanged after both runs.
Vision and idle drivers are outside this first isolated measurement.
Configured context capacity is not an exercised 16k-token prompt.

The resident is asked to consult its auxiliary on whether retrying a 10%-failure
operation always reduces failure to 1%, including correlated server outages.
It chooses the actual `request_deep_analysis` call and writes the tool arguments.
The tool contacts the separate native endpoint and returns GLM's text; the same
resident then responds and handles a fresh arithmetic request.

The existing standalone endpoint option is used. Its provenance honestly reports
`external-unverified` rather than borrowing EIE's Job-owned launch attestation.
The native process and artifacts are identified by the experiment's launch record.
The deep HTTP timeout is extended only in the experimental config, not production.

## Reproduce the native probe independently of Next

Build the native source revision above, not EIE's pinned Gemma runtime:

```powershell
cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF
cmake --build build-cuda --config Release --target llama-server -j4
$env:NVIDIA_TF32_OVERRIDE = '0'
# Replace this value with shard 00001's path; all six shards must be adjacent.
$glmModelPath = 'D:\models\GLM-5.3-Flash-UD-Q4_K_XL-621d456e\UD-Q4_K_XL\GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf'
.\build-cuda\bin\Release\llama-server.exe -m $glmModelPath --alias glm-5.3-flash-native --host 127.0.0.1 --port 18280 --load-mode mmap --no-repack --fit off --no-warmup --flash-attn off -ngl 20 --cpu-moe -c 2048 -b 32 -ub 1 -np 1 -t 8 -tb 8 --metrics --jinja --reasoning-budget 0
```

POST this JSON to `/v1/chat/completions` after `/health` returns HTTP 200:

```json
{"model":"glm-5.3-flash-native","messages":[{"role":"user","content":"Combien font 17 + 25 ? Reponds uniquement avec le nombre."}],"temperature":0,"seed":0,"max_tokens":16,"stream":true,"stream_options":{"include_usage":true},"chat_template_kwargs":{"enable_thinking":false}}
```

## Boundaries and implications for EWS

- Native CPU-expert mmap execution works on this particular laptop. It would be
  false to claim EWS is the only way to make GLM generate here.
- The OS file-backed working set exceeded 24 GB. This is not a bounded EWS cache,
  nor proof of a comfortable RAM budget with arbitrary desktop workloads.
- No GLM expert route trace, payload-byte counter, physical PCIe measurement,
  energy saving, broad quality test, cold/warm timing series or long-context run.
- The Next envelope is not independently replayable without Next. The native
  loading/arithmetic probe above does not depend on the proprietary application.
- EWS still needs GLM runtime integration, six-shard addressing, separate gate/up/down
  tensors, 42 routed trunk layers, 288 logical experts and correct slot remapping.
  Preserve logical router IDs and exclude MTP tensors. Share placement with the 12B.
- The earlier Next offline false-attribution finding is not retested or resolved
  by successful online GLM calls.

This baseline establishes a working reference for the EWS port, not its success.
