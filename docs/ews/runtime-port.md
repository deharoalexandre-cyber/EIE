# Consumed EWS path in EIE (experimental)

This port runs Gemma 4 26B-A4B expert matmuls on the expert slabs actually
loaded into a bounded per-layer cache. It is distinct from the older
trace-driven SLRU / SHA-256 experiment. This pinned runtime recipe is for Gemma.
GLM uses a [separate experimental runtime and patch](../GLM_EWS_EXPERIMENT.md),
not this submodule revision. Do not apply the two patches on top of each other.

Measured outcome and limitations: [2026-09-05 integration report](../../experiments/ews_target/RESULTS.md).

## Scope

- Pinned llama.cpp: `2168b0cd8b87c75c29a1e6588692ebbb805b9bd2` plus
  `patches/ews-runtime-2168b0.patch`.
- Gemma4 26B-A4B, 128 experts/layer, top-8, fused gate/up weights.
- `ews_slots` belongs to a model alias: `0` means normal loading, `8..127`
  means streaming. Other model aliases are unchanged.
- One-token microbatches for both prefill and decode. This trades prefill
  speed for a bounded active expert set; it is not batched serving.
- Windows direct file reads, no full expert RAM cache and no synchronous
  per-chunk SHA in the inference loop. The non-Windows reader uses the OS
  page cache and has not been validated by this Windows campaign.
- Cache state belongs to the model and survives serialized chat / one-shot
  context changes. Inference errors propagate through HTTP/SSE.
- Scales and LoRA expert selection keep logical IDs; only weight matmuls
  receive physical slot IDs. LoRA itself is not covered by the campaign.

## Build

From an initialized EIE checkout at the pinned submodule revision:

```powershell
git -C llama.cpp apply ../patches/ews-runtime-2168b0.patch
cmake -S . -B build-ews -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_MTMD=OFF -DLLAMA_OPENSSL=OFF -DEIE_BUILD_EWS_TESTS=ON
cmake --build build-ews --config Release --target eie-server ews-forward -j 6
```

Apply the patch once; an already-patched checkout does not need it again.
The patch also fixes a duplicate `extern` declaration blocking MSVC shared
builds. EIE links the pinned runtime's compiled `cpp-httplib` library explicitly.

## Run alongside Next

Start Next normally, and run the 26B in a separate EIE process. Do not replace
Next's current 12B server with this binary. This is coexistence, not automatic
delegation of Next's turns to the 26B.

The paragraph above describes the initial coexistence milestone. The subsequent
local Next integration adds `request_deep_analysis` to its existing tool circuit
and starts this sidecar after its resident, using a 4096-token F16 configuration.
Next's `docs/approfondissement_ews.md` records that separate end-to-end campaign,
including its first failed offline-recovery attempt. Do not start a second EIE
on port 18080 when using that updated Next launcher.

For such auxiliary requests, `strict_model: true` disables fallback to a different
loaded alias, and `truncate_prompt: false` requires room for the intact prompt
plus the requested output budget. Overflow returns `context_length_exceeded`
(an SSE error code for streamed requests). Legacy defaults are unchanged.
The API now distinguishes `finish_reason: length` from a normal `stop`.

Example configuration (replace the model path):

```yaml
host: 127.0.0.1
port: 18080
auto_discover: false
type_k: f16
type_v: f16
n_ctx: 2048
flash_attn: true
models:
  gemma-26b-ews: C:/models/google_gemma-4-26B-A4B-it-Q4_0.gguf
ews_slots:
  gemma-26b-ews: 16
preload: [gemma-26b-ews]
```

```powershell
$env:PATH = "$PWD/build-ews/bin/Release;" + $env:PATH
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
./build-ews/Release/eie-server.exe --config ./ews-local.yaml
```

CUDA graphs are disabled for the measured EIE process only; Next retains its
normal graph settings. The flag is a measured runtime profile, not a claim
that all other profiles are incorrect. The endpoint is
`POST /v1/chat/completions` with model `gemma-26b-ews`. `one_shot: true`
uses an ephemeral context while keeping the persistent chat KV intact.

`GET /v1/admin/ews/status` returns actual callback/cache/read counters.
`GET /v1/admin/vram/status` returns the device API's memory figures per model
(shared devices must not be summed). Use external GPU sampling as well;
Windows residency/budget accounting can differ from driver allocation figures.
The OpenAI `/v1/models` catalog remains unchanged.

## Reproduce the numerical gate

The supplied runner is Windows-specific (Release executable/DLL paths) and
requires Python with NumPy installed. The recorded runtime pin is historical
metadata, not a runtime assertion; retain the actual source/build/model hashes
from your own run. The full-reference arm requires enough free VRAM.

```powershell
python experiments/ews_target/run_forward.py --model C:/models/google_gemma-4-26B-A4B-it-Q4_0.gguf --out build-ews/forward-new --predict 32
```

This launches full-weight references before streaming candidates, so do it
with Next stopped. Both arms have identical one-token scheduling, router
callback boundaries, F16 cache and disabled CUDA graphs. All vocabulary logits
are compared as FP32 bit patterns, not just final text. Output directories must
be new to retain previous results. Two small prompts do not establish broad
quality or long-context numerical equivalence.

`experiments/ews_target/next_envelope.py` instead composes the real Next code
with a separately copied `memory`, `proofs` and `user-files` state. It preserves
16K configured context, Nomic, the vision attempt and enabled lifecycle drivers.
It records actual tool calls, not just the model's claims about using tools.
Raw outputs contain private state-derived content and belong under ignored
`build-*` directories. Production state hashes are checked before and after.

The EIE KV-name mapping also now selects `TURBO2_0/TURBO3_0/TURBO4_0`, not
the similarly named model-weight formats. The numerical EWS gate above uses
F16 deliberately; it does not certify TurboQuant numerical equivalence.
