# Experimental GLM-5.3-Flash EWS port

This is a separate runtime path, not a replacement of EIE's pinned Gemma runtime.
The [native laptop run](benchmarks/glm53-native-next-20260908.md) established
CPU/mmap feasibility; it did not establish EWS. Keep those measurements separate.

## Implementation

- Runtime base: `unslothai/llama.cpp`, commit
  `b9b8207fcfc2962093b9466df7af4ff29c2a81ef` (experimental `glm5next` support).
- Compact gate/up/down tensors use physical cache slots. Routing weights, biases
  and scales continue to index logical experts, not cache locations.
- The reader follows split-GGUF metadata, including a metadata-only first shard.
  It reads each selected expert's actual quantized slab; it does not assume a
  uniform Q4 slab size. Gemma's fused gate/up representation remains supported.
- MTP is off. For this artifact the stream covers trunk layers 3 through 44:
  42 layers, 288 experts, top-8 routing, three projections per expert.
- EIE's existing answer-only mode closes an open reasoning prefix using the
  native template's delimiters. The GLM GGUF template ignores `enable_thinking`
  and otherwise emits an open `<think>` despite that setting. This is chat
  formatting, not an added instruction to the model or a larger token allowance.
- Native reference and initial candidate both compute routed experts on CPU,
  with `gpu_layers: 20` for other weights. This isolates streaming from a change
  in CPU/GPU arithmetic. It is not an all-GPU expert acceleration claim.

The model's routed trunk tensors total 185,478,414,336 bytes. An 8-slot cache
contains 5,152,178,176 tensor bytes. These are tensor payloads, **not** process
RAM/VRAM peaks. Other weights, KV/state, scratch, alignment and the resident
model must be counted separately.

## Measurement profile

Use F16 KV, flash attention off, MTP off, 8 CPU threads, one-token microbatches,
`NVIDIA_TF32_OVERRIDE=0`, `GGML_CUDA_DISABLE_GRAPHS=1`. The native arm uses mmap;
EWS allocates compact tensors and explicitly reads slabs. Latency is measured,
not used as a feasibility threshold. No cold-cache protocol is claimed.

`/v1/admin/ews/status` reports:

- `payload_bytes`: slab bytes explicitly installed in the expert cache;
- `host_payload_bytes` / `device_payload_bytes`: destination buffer breakdown;
- `host_expert_bytes` / `device_expert_bytes`: bound cache tensor allocation,
  distinguished from the cumulative copy counters above;
- `read_bytes`: reader bytes including aligned direct-I/O padding on Windows;
- logical and physical expert tensor bytes, callbacks, hits and misses.

Host copies are not GPU transfers. These counters do not measure all PCIe
traffic, operating-system traffic, SSD-controller misses, electricity or energy.

## Qualification status

The GLM forward smoke gate passed locally: 6 prompt tokens and 4 predicted
tokens, 619,520 logits, native versus 8 slots. All compared float bit patterns
and generated token IDs matched, maximum absolute difference 0. This is one
short prompt, not an all-input equivalence result. The EWS counters recorded
378 callbacks, 793 hits, 2,231 misses, 34,241,445,888 payload bytes installed in
host buffers, and 34,268,860,416 reader bytes. Device payload was zero in this
CPU-expert profile.

The [actual virgin-Next EIE/EWS integration](benchmarks/glm53-ews-next-20260908.md)
also passes its functional gate: 12B tool call, complete 702-token GLM answer
(`stop`, not truncated), resident synthesis and a separate resident continuation.
The auxiliary took 1,090.547 seconds. The answer was much longer than requested;
this is not a quality or speed benchmark. The first launcher failure and the
second, truncated reasoning output remain in the receipt.

The updated reader passed the original pinned Gemma runtime's French/code
forward checks (8 generated tokens each; 8 and 32 slots) with bit-identical
logits. Synthetic split-reader/LRU/direct-I/O and per-model configuration tests
also passed (46 assertions, no model or GPU). This does not
substitute for the GLM gate.

## Reproduce in a separate checkout

Start with a **new** EIE clone; do not modify a running Next installation or
apply this patch over the Gemma runtime patch. Keep the 199.7 GB, six-shard
artifact from the native receipt unchanged and use its first shard as `MODEL`.

```powershell
git submodule update --init llama.cpp
git -C llama.cpp fetch https://github.com/unslothai/llama.cpp.git b9b8207fcfc2962093b9466df7af4ff29c2a81ef
git -C llama.cpp checkout --detach b9b8207fcfc2962093b9466df7af4ff29c2a81ef
git -C llama.cpp apply ../patches/ews-runtime-glm5next-b9b8207.patch
cmake -S . -B build-ews -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF -DEIE_BUILD_EWS_TESTS=ON -DEIE_BUILD_SERVING_TESTS=ON
cmake --build build-ews --config Release --target eie-server ews-forward ews-stream-unit
python experiments/ews_target/run_forward.py --model MODEL --out build-ews/new-glm-run --profile glm-cpu --slots 8 --predict 4
```

The runner needs NumPy and records its binary hashes, runtime revision/diff,
raw logits and exact comparison. Output directories must be new. Windows runs
need `build-ews/bin/Release` on `PATH` when starting executables directly.
Architecture 89 is the measured RTX 4090 Laptop profile, not a universal target.

For EIE serving, use `preload` with the model alias and this configuration:

```yaml
host: 127.0.0.1
port: 18280
strategy: generic
auto_discover: false
type_k: f16
type_v: f16
flash_attn: false
n_ctx: 2048
preload: [glm-5.3-flash-ews]
models:
  glm-5.3-flash-ews: PATH_TO_FIRST_GGUF_SHARD
ews_slots:
  glm-5.3-flash-ews: 8
gpu_layers:
  glm-5.3-flash-ews: 20
cpu_moe:
  glm-5.3-flash-ews: true
threads:
  glm-5.3-flash-ews: 8
```

Set `NVIDIA_TF32_OVERRIDE=0` and `GGML_CUDA_DISABLE_GRAPHS=1` before launching
`build-ews/Release/eie-server.exe --config PATH_TO_CONFIG`. This runtime does
not provide TurboQuant KV types. The original pinned Gemma runtime still does;
the compatibility adapter does not relabel F16 as TurboQuant.

After loading, an ordinary HTTP client can exercise the EIE/GLM path without
Next. For example, POST the following JSON to `/v1/chat/completions` and retain
the response plus `/v1/admin/ews/status` before and after it:

```json
{"model":"glm-5.3-flash-ews","messages":[{"role":"user","content":"An operation fails in 10 percent of cases. Does retrying it once always reduce failure to 1 percent, even when both attempts depend on the same unavailable server? Explain briefly."}],"temperature":0,"max_tokens":768,"stream":false,"strict_model":true,"truncate_prompt":false,"one_shot":true}
```

Allow the request to finish; this feasibility profile has no speed threshold.
Record `finish_reason` and any error rather than counting a partial answer as a
complete result. This plain HTTP example exercises EIE, not Next's resident/tool
loop, and its prompt is not identical to the resident-authored French question.

## Next qualification steps

1. Repeat the paired numerical test with a longer, frozen multi-domain corpus.
   Separate prefill from decode, and reserve a holdout split before tuning.
2. Expand the [bounded hybrid GPU placement](benchmarks/glm53-ews-gpu-next-20260908.md).
   It has a real Next roundtrip, a two-layer native GPU control and three
   larger-placement cache consistency checks; not all-GPU expert qualification.
3. Measure cache-hit distributions and bytes per generated token against
   explicit RAM/VRAM budgets before adding prefetch or a second cache tier.
4. Repeat native/EWS timing under a declared cache protocol. Count complete
   request time and whole-system memory pressure, not just expert-cache bytes.
5. Exercise long Next context, auxiliary failures and lifecycle activity on
   copied state. Keep the earlier offline-attribution failure visible.

No automatic production replacement, new permission layer, or per-chunk
cryptographic gate is part of this port.

## GPU placement reproduction

The [GPU experiment and live receipt](benchmarks/glm53-ews-gpu-next-20260908.md)
keep the runtime patch, 8-slot cache and F16 profile. It
changes `cpu_moe` to `false`, allowing the normal layer placement to put experts
on CUDA. For this runtime, `gpu_layers: 20` counts output and the unused MTP
position too: **18 routed layers (27..44) use CUDA; 24 (3..26) use CPU**.
This is hybrid streaming, not all 42 expert layers on GPU.

With 8 slots, the bound expert tensors contain 2,198,339,584 device bytes and
2,953,838,592 host bytes. Check those fields and the cumulative device payload
in `/v1/admin/ews/status`; occupied VRAM alone does not prove GPU expert use.

Run numerical checks without the Next resident first:

```powershell
# A native GPU reference that fits: routed layers43/44 plus output on CUDA.
python experiments/ews_target/run_forward.py --model MODEL --out build-ews/new-gpu-native --profile glm-gpu --gpu-layers 4 --slots 8 --predict 8
# The larger placement: cache-size consistency, NOT a native reference.
python experiments/ews_target/run_forward.py --model MODEL --out build-ews/new-gpu-smoke --profile glm-gpu --gpu-layers 20 --reference-slots 16 --slots 8 --predict 8
python experiments/ews_target/run_forward.py --model MODEL --out build-ews/new-gpu-code --profile glm-gpu --gpu-layers 20 --reference-slots 16 --slots 8 --predict 8 --prompt experiments/ews_target/glm53-gpu-code-prompt.txt
python experiments/ews_target/run_forward.py --model MODEL --out build-ews/new-gpu-fr --profile glm-gpu --gpu-layers 20 --reference-slots 16 --slots 8 --predict 8 --prompt experiments/ews_target/glm53-gpu-fr-prompt.txt
```

Native full expert tensors at `gpu_layers: 20` do not fit this GPU. Do not call
the 16-vs-8-slot comparison native equivalence. Both caches can share an error;
the separate native GPU control covers only two routed layers and one prompt.
Likewise, CPU and GPU placements may differ numerically; equality is tested
within each fixed placement, not across them.

The first probe used `gpu_layers: 1`: only output was offloaded, device expert
payload was zero. It passed numerically but did **not** qualify GPU experts.
That probe is retained rather than relabeled as a GPU success.
