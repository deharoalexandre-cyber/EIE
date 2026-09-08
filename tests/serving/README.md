# Serving functional regression tests

## Offline unit tests

```bash
cmake -S tests/serving -B build-serving
cmake --build build-serving --config Release
ctest --test-dir build-serving -C Release --output-on-failure
```

No models or GPU. Tests exercise the actual output helper, response serializer,
loaded-model registry and concurrent metrics. Every byte partition of two stop
fixtures is checked. They do not prove the model tokenizer or GPU inference.

## Real HTTP, fake model

Supply the pinned runtime's vendor directory and its compiled cpp-httplib
library (no llama library is linked):

```powershell
cmake -S tests/serving -B build-serving -DEIE_HTTP_VENDOR=C:/path/to/llama.cpp/vendor -DEIE_HTTP_LIBRARY=C:/path/to/cpp-httplib.lib
cmake --build build-serving --config Release
python tests/serving/http_contract.py build-serving/Release/eie-http-fixture.exe
```

The Python standard-library runner starts only its own localhost fixture on a
temporary port and stops that process at the end. It exercises the real EIE
routes, SSE, UTF-8, usage, injected errors/recovery and concurrent requests.
Its predetermined prompt count is a serializer check, not tokenizer evidence.

## Output and usage semantics

- The first completed nonempty stop sequence ends generation, even inside a
  token piece. Among stops completing at the same byte, the earliest start wins.
- Only a suffix that can still become a stop and incomplete UTF-8 bytes are
  withheld. A partial unmatched stop is emitted at normal EOS/length completion.
- An incomplete final UTF-8 character becomes U+FFFD consistently in both modes.
- Client cancellation stops callbacks; unresolved buffered bytes are not flushed.
- `prompt_tokens` is the retained tokenized prompt including BOS/template and
  any reused prefix. After permitted truncation, discarded tokens are not counted.
- `completion_tokens` counts sampled model tokens, including a terminal EOG
  or a token containing the stop text; it is not a retokenization of visible text.
- `prompt_tokens_details.cached_tokens` is the reused prefix, zero for one-shot.
- Health and the loaded-model gauge come from the model registry, not prior
  request activity. This is not a successful inference-readiness probe.

Run the separate real-model gate for a new build; an offline pass alone does not
qualify inference. The [8 September receipt](../../docs/benchmarks/serving-functional-20260908.md)
records executed 12B and streamed 26B passes, plus the distinct failed Next
offline-attribution scenario. Do not convert a native serving pass into a claim
that an agent always calls or correctly attributes its tools.

Build with `-DEIE_BUILD_SERVING_TESTS=ON` in an initialized/patched full checkout.
`serving-model-contract MODEL_GGUF EWS_SLOTS` exercises the actual backend and
tokenizer: use 0 for normal loading or 16 for the known Gemma EWS profile.
It loads a model and can use the GPU; run only in the dedicated test window.
The offline CTest does not launch this executable automatically.

## Real HTTP, real models (Windows/CUDA profile)

After building the full initialized/patched checkout:

```powershell
python tests/serving/real_http_contract.py --eie C:/path/to/EIE --out C:/path/to/new-test-output --model C:/models/gemma-4-12B-it-QAT-Q4_0.gguf --embedder C:/models/nomic-embed-text-v2-moe.Q6_K.gguf
```

This opt-in test loads both models with F16 KV, 512 context tokens, and disabled
CUDA graphs. It starts `build-ews/Release/eie-server.exe`, puts its freshly built
`build-ews/bin/Release` libraries on PATH, uses a temporary localhost port, and
stops only that process. The output directory must not already exist.
Assertions cover chat/SSE/stop parity, usage, two finite nonzero 768-dimensional
embeddings, catalog/health/metrics, HTTP 404/400 and next-request recovery.
The dimensions and short prompt are specific to this known-model profile;
this is not a generic all-model quality test. It does not run production Next.
