# Runtime authentication, routing measurement and recovery

Source update: 14 September 2026. These changes do **not** modify already
published Mac, Windows or Android bundles. Rebuild from the updated source to
use them. No production Next configuration or executable is changed by this work.

## Optional authentication

`auth_token` is now enforced, when non-empty, on the API, administration,
health and metrics routes. Send `Authorization: Bearer <token>`.
Missing, wrong, malformed or duplicate credentials return HTTP 401 with an
explanation and `WWW-Authenticate: Bearer`. POST authentication runs after the
HTTP library consumes the body, but before application parsing or inference.
This avoids a Windows connection reset on early rejection of a fragmented POST.

With no configured token, local behavior is unchanged. The minimal preset
reader accepts an **unquoted** value; a random hexadecimal token avoids its
comment/quoting limitations. EIE does not add TLS: use loopback or appropriate
transport protection when sending a bearer token over a network. Old binaries
that only parsed this setting still provide no authentication.

## Opt-in EWS routing measurements

For one request, add `"ews_trace": true` to the chat JSON. To trace all requests
for a particular EWS model, use a preset map:

```yaml
ews_slots:
  auxiliary: 8
ews_trace:
  auxiliary: true
```

The request flag and model default are ORed. The option is off by default.
Retrieve `GET /v1/admin/ews/routing` after the request. It returns the last
request snapshot per model, including a model-local `request_sequence`,
enabled/supported flags and outcome. It waits for that model's inference mutex;
it is **not** a live progress stream. Use an isolated, single-client measurement
session and save each snapshot before another request replaces it. A subsequent
untraced request clears the previous histogram. Failures/cancellation retain
partial counts and an explicit non-complete outcome.

Counts use **logical expert IDs before slot remapping**, including cache hits:

- Per layer and phase (`prefill`, `decode`): selected, hit and miss counts for
  each expert, callback count, first accesses and reuse-distance histogram.
- Expert weight bytes per layer come from the actual GGUF slabs (all relevant
  projections). They exclude non-expert weights, KV, scratch and driver memory.
- Selected = hits + misses = top-k x callbacks on a complete trace. These are
  **hard top-k selection counts**, not router probability mass or quality scores.
- Actual hit/miss counters follow the existing EWS cache; tracing does not
  select experts, change eviction or change the weights that are consumed.

Reuse distance counts distinct intervening experts, separately for each layer.
The recency stack resets at the request boundary and carries across the
prefill/decode boundary. `first_accesses` means first in this trace, **not** a
physical cache miss: a warm model may already contain that expert. For an empty
serial LRU, a repeated access is a hit at capacity C exactly when distance < C.
EWS protects the whole current top-k during victim selection, so that theoretical
curve is **not an exact prediction of EWS hits**. Capacities below top-k are not
deployable EWS configurations. Neither curve predicts multi-tier I/O latency.

The collector uses bounded memory O(layers x experts), independent of conversation
length, and O(experts) recency maintenance per selected expert. It stores no prompt
text or token text and does no per-token disk logging. Measure instrumentation
overhead on the target workload; one timing pair is not a performance benchmark.

### Reproducible first campaign

The [pilot workloads](../tests/routing_workloads_v1.json) contain five domains:
code, writing, math, general conversation and document review. Each has a
calibration prompt and a held-out test prompt. This is an explicitly small,
raw-completion, short-context pilot, **not a long-context Next campaign**.

Build `ews-forward` with the appropriate matched runtime. The optional final
argument `1` enables tracing; `0` leaves it off. With an existing Windows shared
runtime build, [tests/ews_linked](../tests/ews_linked/CMakeLists.txt) builds the probe
without modifying that runtime. Matching headers and libraries are required.

```text
python scripts/routing_campaign.py --binary PATH/ews-forward.exe --model FIRST_SHARD.gguf --runtime-dir RUNTIME_DLL_DIR --model-receipt MODEL_RECEIPT.json --output NEW_DIRECTORY
python scripts/analyze_routing.py NEW_DIRECTORY
```

The runner freezes the workload, sampling, binary/library hashes and conditions
before inference. It retains stderr, stdout, logits, JSON counters and failures.
The supplied model provenance is retained but its shard hashes are **not silently
claimed to have been rechecked**. Each item starts with a new process and empty
EWS/KV caches; OS cache state is not controlled. The first item runs with tracing
off/on and must have byte-identical logits before the campaign continues.

Analysis separates prefill/decode and domains. It reports in-sample concentration
(an optimistic oracle) separately from held-out coverage using calibration-ranked
experts. Uniform slots-per-layer curves include expert weight bytes. A fixed
ranking that works only in its calibration domain is not a universal hotset.

Before a deployment-size or model-family conclusion, expand the **pre-frozen**
corpus across 1k/4k/16k/64k contexts, multiple prompts and repeats per domain,
measure overhead, cross-domain transfer and real Next sessions. GLM results say
nothing about Kimi's distribution; that model needs its own matched campaign.

## Bounded group recovery

`fallback: retry_once` permits at most one extra call for a failed member.
`fallback: replace_with` invokes the loaded alias named by `replacement:` or
`replace_with:` once for that failed member. No implicit loading or retry loop
is added. These explicit settings apply to all four built-in strategies.

In parallel groups, successful members are not rerun; recovery stops once the
response quorum is met. In sequential chains, a failed step must recover before
its result is passed downstream. The replacement uses its own chat template.
Cancellation and an error after streamed output do not trigger another generation.
An unloaded replacement produces a named error, not a false success.

`responses` holds final member/step outcomes (the selected answer for fan-out).
`attempts` retains initial failures and recovery attempts, with requested alias,
actual alias, action, error, content and finish reason. `completed` counts
successful final steps, not attempts. Sequential `required` is the chain length.
A reused backup is not an independent vote; fan-out still picks the longest
successful response, not the best-quality answer. This does not change the
separate, unresolved Next behavior of attributing an answer to an uncalled tool.

## Local checks and remaining scope

- 628 existing no-model serving assertions; 157 recovery assertions using the
  production scheduler with injected failures, exceptions and cancellation.
- 41 routing assertions, including 10,000 seeded accesses checked against
  independent direct LRU simulations at 32 capacities.
- 4,726 assertions through the real EWS reader/callback with small synthetic
  split GGUF tensors: hits, misses, high IDs, phase split, conservation and
  identical consumed tensor bytes with tracing off/on. Not a trained-model run.
- Real HTTP handlers with a deterministic fake backend: 12 serving scenarios,
  60 concurrent requests, 4 recovery scenarios, trace flag/reset and 139 auth
  checks, including delayed POST bodies, duplicate headers and authorized SSE.
- The full standard EIE runtime builds locally. The experimental GLM probe is
  built against its separate matching runtime. Actual pilot measurements are
  recorded separately; these test counts are not fleet reliability claims.
- The [real GLM pilot](benchmarks/glm53-routing-20260914.md) passes all ten items
  and its off/on logit comparison. A separate real `CpuBackend` test passes trace
  completion, disabling/reset, output equality, overflow, cancellation and recovery.

`reserve_mb`, automatic KV selection, dynamic eviction and audit-chain integrity
remain outside this change and retain their existing limitations.
