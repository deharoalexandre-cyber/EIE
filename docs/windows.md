# EIE on Windows: prebuilt CPU bundle (x64)

One bundle is published on the [releases page](https://github.com/deharoalexandre-cyber/EIE/releases):

| Bundle | Target | Backend | Status |
|---|---|---|---|
| `eie-windows-x64-<rev>.zip` | Windows 10/11 x64, any CPU with AVX2/FMA/F16C (Intel Haswell 2013+, AMD Zen) | CPU only, no CUDA | see the [Windows CPU receipt](benchmarks/windows-cpu-20260914.md) |

The bundle contains a **static `eie-server.exe`** (no DLL, no Visual C++ redistributable,
no OpenSSL: the server speaks plain HTTP on `127.0.0.1`), the `windows-cpu.yaml` and
`generic.yaml` presets, `start-eie.bat`, `build-info.txt` (compiler and CMake options
used), `SHA256SUMS` and this file. Model weights are **not** bundled.

This is the consumer-CPU line of the repository: a laptop-class processor without a
GPU. It is distinct from the Windows/CUDA build used by the EWS campaign (which is built
from source, see the [README](../README.md#windows-cuda)) and from any server-class CPU
receipt.

The labels used in the receipts follow the repository's [claims audit](CLAIMS_AUDIT.md):
*locally verified* means a JSON receipt produced by `scripts/receipt-windows.py` on that
machine is retained under `benchmarks/data/`; *maintainer-reported* means figures were
relayed without that receipt.

## Install

1. Unzip `eie-windows-x64-<rev>.zip` anywhere (no administrator rights needed).
2. Check the hashes: `certutil -hashfile eie-server.exe SHA256` and compare with `SHA256SUMS`.
3. Drop your GGUF files into `models\`: one chat model, and one embedding model whose file
   name contains `bge` or `embed` if you need `/v1/embeddings`.
4. Double-click `start-eie.bat`, or run it from a terminal. The engine listens on
   `http://127.0.0.1:8090`; `curl http://127.0.0.1:8090/health` reports the loaded models.

No service or scheduled task is installed. To start the engine with the session, put a
shortcut to `start-eie.bat` in `shell:startup`.

Windows SmartScreen may warn on first launch because the executable is not code-signed:
choose *More info* then *Run anyway*, or unblock the file in its Properties.

## Presets

| Preset | KV cache | Notes |
|---|---|---|
| `presets/windows-cpu.yaml` | f16 | `preload: all`, `n_ctx 4096`, threads = half the logical processors (engine default) |
| `presets/generic.yaml` | engine default | reference preset shared with the other platforms |

**Always start the engine with `--config`.** Without a preset the engine uses its built-in
defaults (`type_k/type_v = turbo3`, port 8080, `model_dir /models`). Set a `threads:`
map in the preset to override the automatic thread count per model alias.

## Verify a machine (produce a receipt)

```powershell
python scripts\receipt-windows.py --bundle C:\path\to\eie-windows-x64-<rev> --models C:\path\to\models --port 8099 --rev (git rev-parse --short HEAD) --out receipt.json
```

The script starts **its own** engine instance on the test port (never benchmark the
instance someone is talking to), then records: load time until every GGUF in the models
directory is healthy, an embedding, a cold-prefix chat, a warm-prefix chat and a 256-token
warm-prefix generation: with `prompt_tokens`, `completion_tokens`, `cached_tokens`, the
server-side `[KV]` time, and queue/transport time separated from engine time. It also
records the CPU, memory, Windows build, the effective KV type, context and thread count
per model, the toolchain from `build-info.txt`, and the SHA-256 of the binary and of
every GGUF. Python 3.8+ from python.org is enough; no third-party module. The JSON goes
next to the Markdown receipt in `docs/benchmarks/data/`.

## Build from source

```powershell
git clone --recursive https://github.com/deharoalexandre-cyber/EIE.git
cd EIE
git -C llama.cpp apply ..\patches\ews-runtime-2168b0.patch     # required since 7 September 2026
scripts\build-windows-cpu.bat                                   # -> build-windows-cpu\eie-server.exe
python scripts\bundle-windows.py                                # -> dist\eie-windows-x64-<rev>.zip
```

`build-windows-cpu.bat` needs `cmake` and `ninja` on `PATH` plus either GCC (MinGW-w64;
`winget install BrechtSanders.WinLibs.POSIX.UCRT` installs it for the current user
without administrator rights, together with `cmake`) or MSVC from a Developer PowerShell
for VS 2022. The published bundle is the GCC/static profile; `build-info.txt` in each
bundle names the exact compiler. `GGML_NATIVE=OFF` keeps the binary portable across CPU
generations, at the cost of AVX-512 on the machines that have it.

## Not covered

CUDA, EWS on this build (the code path is compiled in, not qualified), multi-model groups,
Windows on ARM, 32-bit Windows, CPUs without AVX2, any model other than the two files named
in the receipt, running as a Windows service.
