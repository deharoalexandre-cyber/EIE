#!/usr/bin/env python3
"""Isolated real-model receipt for EIE on Windows HIP (standard library only).

Run one GGUF at a time with synthetic French input. This tool starts and stops
its own loopback EIE server; it does not connect to an existing service.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get_json(url: str, *, timeout: int = 2) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post_json(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kind", choices=("chat", "embedding"), required=True)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--dll-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-backend", choices=("rocm", "cpu"), default="rocm")
    parser.add_argument("--startup-timeout", type=int, default=300)
    args = parser.parse_args()
    for path in (args.exe, args.model, args.preset):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    if not args.dll_dir.is_dir():
        parser.error(f"missing DLL directory: {args.dll_dir}")
    if args.startup_timeout <= 0:
        parser.error("startup timeout must be positive")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".log")
    model_alias = args.model.stem
    port = free_port()
    command = [str(args.exe.resolve()), "--config", str(args.preset.resolve()),
               "--model", str(args.model.resolve()), "--host", "127.0.0.1",
               "--port", str(port)]
    env = dict(os.environ)
    extra_paths = [str(args.dll_dir.resolve()), str(args.exe.resolve().parent)]
    if env.get("HIP_PATH"):
        extra_paths.append(str(Path(env["HIP_PATH"]) / "bin"))
    env["PATH"] = os.pathsep.join([*extra_paths, env.get("PATH", "")])

    receipt = {
        "schema": "eie.windows-hip-isolated/v1",
        "scope": "real EIE executable, one GGUF, synthetic prompt, loopback",
        "utc": datetime.now(timezone.utc).isoformat(),
        "exe": str(args.exe.resolve()),
        "exe_sha256": digest(args.exe),
        "model": str(args.model.resolve()),
        "model_sha256": digest(args.model),
        "model_alias": model_alias,
        "kind": args.kind,
        "expected_backend": args.expected_backend,
        "command": command,
        "log": str(log_path.resolve()),
        "port": port,
        "trials": [],
        "ok": False,
    }
    process = None
    started = time.monotonic()
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command, cwd=str(args.dll_dir.resolve()), env=env,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"EIE exited during load: {process.returncode}")
                try:
                    health = get_json(base + "/health")
                    if health.get("models") == 1:
                        receipt["health"] = health
                        break
                    if health.get("models") == 0:
                        raise RuntimeError("EIE started without the requested model")
                except (OSError, urllib.error.HTTPError, ValueError):
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError("EIE did not report one loaded model")
            receipt["startup_seconds"] = round(time.monotonic() - started, 3)
            try:
                receipt["vram_status"] = get_json(base + "/v1/admin/vram/status")
            except (OSError, urllib.error.HTTPError, ValueError) as exc:
                receipt["vram_status_error"] = f"{type(exc).__name__}: {exc}"

            endpoint = "/v1/embeddings" if args.kind == "embedding" else "/v1/chat/completions"
            body = ({"model": model_alias, "input": "Essai local de recherche documentaire."}
                    if args.kind == "embedding" else
                    {"model": model_alias,
                     "messages": [{"role": "user", "content": "À quoi sert un rappel de rendez-vous ? Réponds en français, en une phrase."}],
                     "max_tokens": 96, "temperature": 0, "stream": False})
            for trial_number in (1, 2):
                request_started = time.monotonic()
                response = post_json(base + endpoint, body)
                trial = {"trial": trial_number,
                         "wall_seconds": round(time.monotonic() - request_started, 3)}
                if args.kind == "embedding":
                    vector = response["data"][0]["embedding"]
                    trial["dimensions"] = len(vector)
                    trial["ok"] = len(vector) == 768
                else:
                    choice = response["choices"][0]
                    content = choice["message"].get("content") or ""
                    trial.update({"content_excerpt": content[:300],
                                  "finish_reason": choice.get("finish_reason"),
                                  "usage": response.get("usage"),
                                  "ok": bool(content.strip())})
                receipt["trials"].append(trial)
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log_path.exists():
            log = log_path.read_text(encoding="utf-8", errors="replace")
            receipt["backend_device_lines"] = [line[:500] for line in log.splitlines()
                                               if re.search(r"ROCm0|offloaded \d+/\d+ layers|\[ROCm\]", line)][:80]
            match = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
            receipt["gpu_layers_offloaded"] = ([int(match.group(1)), int(match.group(2))]
                                                 if match else None)
            receipt["rocm_device_in_log"] = "ROCm0" in log
            if "error" in receipt:
                receipt["log_tail"] = log.splitlines()[-25:]
        inference_ok = ("error" not in receipt and len(receipt["trials"]) == 2
                        and all(trial["ok"] for trial in receipt["trials"]))
        offload = receipt.get("gpu_layers_offloaded")
        gpu_ok = (bool(receipt.get("rocm_device_in_log")) and bool(offload)
                  and offload[0] == offload[1])
        receipt["ok"] = inference_ok and (gpu_ok if args.expected_backend == "rocm" else True)
        args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": receipt["ok"], "out": str(args.out),
                          "error": receipt.get("error"),
                          "gpu_layers_offloaded": receipt.get("gpu_layers_offloaded")}, ensure_ascii=False),
              flush=True)
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
