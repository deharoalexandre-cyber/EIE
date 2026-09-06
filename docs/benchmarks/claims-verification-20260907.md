# Publication verification receipt - 7 September 2026

Performed by Codex at Alexandre De Haro's request, on the maintainers' machine.
This is not an independent audit. **No GPU inference, benchmark campaign,
Next restart or production-state modification was performed for publication.**

Initial public base: `07b1500e6b5f68b5fb12a3118aff832d217bbfc6`.
Local EWS source came from the already-tested working tree based on
`67a779d`; the Apple Silicon changes from the public base were retained.
Publication is prepared in an isolated worktree, not by resetting or staging
the live working tree.

## Checks and meaning

| Check | Result | What it does not establish |
|---|---|---|
| September pilot files | SHA-256 and size rechecked for 24 existing JSON/trace/logit files; counters agree with summary | Not a new performance run |
| Pilot comparisons | Five positive comparisons plus one negative control checked against published summary and artifact hashes | Not all-input quality |
| Target runtime logits | Six unique raw logit files rehashed and size-checked; four paired comparisons | Not other cache formats/devices |
| Real Next reports | Two existing raw report hashes match | Private conversations are not published; no broad lifecycle or quality proof |
| August archive | Seven tracked Git blobs match their announced hashes | Raw traces/timings and ten manifest prompt files are missing publicly |
| Diagnostic C++ probe | Nine source-behavior observations, CTest success | Intentionally characterizes incomplete features; not nine working product guarantees |
| Inference-enabled C++ compile/link | Four publication server/backend translation units compiled and linked with MSVC 19.44 against the existing patched runtime libraries | No executable launch; not a clean rebuild of CUDA kernels or an inference rerun |
| Runtime patch applicability | `git apply --check --cached` succeeds against the clean index at the pinned submodule commit | Read-only check; no patch applied to the live submodule |
| Publication consistency | Eight Python files parse, three JSON files parse, Node cancellation harness syntax checks, Markdown file links resolve, no control characters | Syntax/link checks are not execution tests |

Thirteen imported source/harness files match the local source text after
line-ending normalization. The two reconciled files differ only by retaining
the public Apple Silicon CMake/offload changes. Frozen August data files are
not edited. No models, executables or private conversation reports are staged.

One initial probe expectation incorrectly assumed the no-llama CUDA/HIP
placeholders now returned zero. They still return hardcoded capacities; the
probe and audit were corrected to record that limitation. The inference-enabled
path separately uses the device-memory API. The final diagnostic run passes.

The diagnostic target compiles without llama.cpp or GPU access and tests
configuration, scheduling, placeholder telemetry and metric accounting.
It must not be reported as a full inference-server build.

Commands (from repository root):

```bash
cmake -S tests/claims -B build-claims
cmake --build build-claims --config Release
ctest --test-dir build-claims -C Release --output-on-failure
python tests/claims/verify_evidence.py
python tests/claims/verify_consumed.py
```

The local raw check additionally supplied the original pilot directory, target
forward directory and Next report build root through the script's three
optional arguments. Without those files the public command does not claim to
verify raw data. No model files or private reports are copied into Git.

## Separate 6 September cancellation evidence

The imported backend also contains the existing disconnect/prefill fix.
The archived local report records the next request's first-token latency after
an SSE prefill disconnect falling from **53.327 s to 2.511 s**, corroborated
by expert callbacks falling from **16,080 to 2,430**.

This is not a universal cancellation deadline: it includes the next request's
own work and cannot interrupt an individual active `llama_decode` call.
RST/persistent/decode cases were separately exercised locally.
The measured Windows executable changed from:

- `0035F4E9F9251D08C589FD71AE158FB7EFA73E0052B8920185070F5F1507AF39`
- to `6B9CECF5CAA207B5A26A4E75A409CE83889BEC8DB37DF7138398D7ADA0218EF2`.

That binary is not distributed here, and a fresh build of the reconciled
publication tree is not asserted to reproduce its executable hash.
The cancellation source and development harness are included; the latter
contains local model-path assumptions and is not a portable turnkey benchmark.

## Reproduction gaps retained rather than hidden

The [August archive notice](data/ews/README.md) explains the resident-weight
timing path, first-observation hashes and missing files. The September report
distinguishes the consumed pilot from the target runtime and the private-state
Next envelope. JSON summaries are not raw tensor downloads.

Clean-clone full builds, repeated performance campaigns, platform qualification
and the remaining serving features are tracked in the
[roadmap](../ROADMAP_TO_CLAIMS.md).
