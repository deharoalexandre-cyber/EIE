# EIE on macOS — prebuilt bundles (Intel and Apple Silicon)

Two bundles are published on the [releases page](https://github.com/deharoalexandre-cyber/EIE/releases):

| Bundle | Target | Backend | Status |
|---|---|---|---|
| `eie-macos-arm64-<rev>.tar.gz` | Apple Silicon (M1 and later), macOS 13+ | Metal, all layers on GPU | see the [Apple Silicon receipt](benchmarks/macos-apple-silicon-20260913.md) |
| `eie-macos-x86_64-<rev>.tar.gz` | Intel Macs, macOS 13+ | CPU (Accelerate); Metal disabled on purpose | see the [Intel receipt](benchmarks/macos-intel-20260914.md) |

Each bundle contains a **static `eie-server`** (no Homebrew, no dylib, no OpenSSL — the
server speaks plain HTTP on `127.0.0.1`), the two macOS presets, `install-macos.sh`,
`SHA256SUMS` and this file. Model weights are **not** bundled.

The labels used in the receipts follow the repository's [claims audit](CLAIMS_AUDIT.md):
*locally verified* means a JSON receipt produced by `scripts/receipt-macos.sh` on that
machine is retained under `benchmarks/data/`; *maintainer-reported* means figures were
relayed without that receipt.

## Install

```bash
tar -xzf eie-macos-*.tar.gz && cd eie-macos-*/
shasum -a 256 -c SHA256SUMS
bash install-macos.sh --download-models      # ~3.7 GB, hashes verified
```

`install-macos.sh` copies the engine and presets into `~/Elyne`, optionally downloads
the reference models (Gemma 4 E2B QAT Q4_0 for generation, bge-m3 Q8_0 for embeddings —
the same files as the receipts, SHA-256 checked), registers a LaunchAgent
(`com.elyne.eie`, port 8090, started with the session, `Interactive` priority) and
waits for `/health`. Bring your own GGUF instead by dropping it into `~/Elyne/models/`.

Gatekeeper: the binaries are ad-hoc signed, not notarized. Files copied from a USB key
usually carry no quarantine flag; if macOS refuses to run `eie-server`, run
`xattr -cr ~/Elyne` or allow it under *System Settings → Privacy & Security*.

## Presets

| Preset | Machine | KV cache | Notes |
|---|---|---|---|
| `presets/macos-silicon.yaml` | Apple Silicon | f16 | all layers offloaded to Metal (engine default 99) |
| `presets/macos-cpu.yaml` | Intel | f16 | Metal is disabled at build time: Intel iGPUs return garbage |

**Always start the engine with `--config`.** Without a preset the engine uses its built-in
defaults (`type_k/type_v = turbo3`, port 8080). The pinned fork has no Metal kernels for
TurboQuant KV types; since `0e8d824` the arm64 build falls back to f16 with a warning, but
the preset is the supported path.

## Verify a machine (produce a receipt)

```bash
bash scripts/receipt-macos.sh --bundle ~/Elyne --models ~/Elyne/models --port 8091 \
     --rev "$(git rev-parse --short HEAD)" --out receipt.json
```

The script starts **its own** engine instance on the test port (never benchmark the
instance someone is talking to), then records: load time until both models are healthy,
an embedding, a cold-prefix chat, a warm-prefix chat and a 256-token warm-prefix
generation — with `prompt_tokens`, `completion_tokens`, `cached_tokens`, the server-side
`[KV]` time, and queue/transport time separated from engine time. It also records the
Metal device and offloaded layers seen in the log, the effective KV type and context per
model, and the SHA-256 of the binary and of every GGUF. The JSON goes next to the
Markdown receipt in `docs/benchmarks/data/`.

## Build from source

```bash
git clone --recursive https://github.com/deharoalexandre-cyber/EIE.git && cd EIE
git -C llama.cpp apply ../patches/ews-runtime-2168b0.patch     # required since 7 September 2026
./scripts/build-macos-arm64.sh                                 # Apple Silicon (also cross-compiles from Intel)
./scripts/build-macos-x86_64.sh                                # Intel (static, portable)
bash scripts/bundle-macos.sh arm64        # or x86_64 → dist/eie-macos-<arch>-<rev>.tar.gz
```

`GGML_NATIVE=OFF` keeps the Intel binary portable across CPU generations; the arm64
build embeds the Metal shader library, so no Xcode Metal toolchain is needed.

## Not covered

EWS on Metal (the code path is compiled in, not qualified), multi-model groups, any
model other than the two reference files, Intel GPUs, macOS 12 or earlier.
