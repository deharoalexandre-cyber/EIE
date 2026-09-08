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

The separate real-model gate must still compare streamed/nonstreamed output,
tokenizer counts, stopped-request KV recovery and resident/auxiliary coexistence
before a candidate is qualified for deployment.

Build with `-DEIE_BUILD_SERVING_TESTS=ON` in an initialized/patched full checkout.
`serving-model-contract MODEL_GGUF EWS_SLOTS` exercises the actual backend and
tokenizer: use 0 for normal loading or 16 for the known Gemma EWS profile.
It loads a model and can use the GPU; run only in the dedicated test window.
The offline CTest does not launch this executable automatically.
