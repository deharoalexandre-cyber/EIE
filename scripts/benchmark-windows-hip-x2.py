#!/usr/bin/env python3
"""Measure isolated EIE HIP chat streams or embedding batches on the X2.

The benchmark uses only synthetic text and starts one model at a time. Its JSON
receipt contains hashes and timings but no local user paths or response bodies.
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
import statistics
import subprocess
import time
import urllib.error
import urllib.request


CHAT_PROMPT = (
    "Redige une liste numerotee de 50 actions concretes pour organiser un bureau. "
    "Chaque ligne contient une phrase differente de six a dix mots. "
    "Continue jusqu'au numero 50."
)
EMBED_INPUTS = [
    f"Document synthetique {index:02d}: suivi du dossier, classement et echeance."
    for index in range(32)
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post_json(url: str, body: dict, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def stream_chat(base: str, alias: str, trial: int) -> dict:
    body = {
        "model": alias,
        "messages": [{"role": "user", "content": CHAT_PROMPT + f" Essai {trial}."}],
        "max_tokens": 384,
        "temperature": 0,
        "stream": True,
    }
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter_ns()
    pieces: list[str] = []
    stamps: list[int] = []
    usage = None
    finish_reason = None
    errors = []
    done = False
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.status != 200 or "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise RuntimeError(f"unexpected streaming response: {response.status}")
        for raw in response:
            stamp = time.perf_counter_ns()
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                done = True
                break
            item = json.loads(payload)
            if "error" in item:
                errors.append(item["error"])
                continue
            if "usage" in item:
                usage = item["usage"]
            for choice in item.get("choices", []):
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                content = (choice.get("delta") or {}).get("content")
                if isinstance(content, str) and content:
                    pieces.append(content)
                    stamps.append(stamp)
    ended = time.perf_counter_ns()
    text = "".join(pieces)
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    wall = (ended - started) / 1e9
    decode_span = (stamps[-1] - stamps[0]) / 1e9 if len(stamps) >= 2 else None
    result = {
        "trial": trial,
        "http_wall_seconds": round(wall, 3),
        "time_to_first_content_seconds": round((stamps[0] - started) / 1e9, 3) if stamps else None,
        "content_piece_count": len(pieces),
        "completion_tokens": tokens,
        "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
        "cached_prompt_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if isinstance(usage, dict) else None,
        "finish_reason": finish_reason,
        "done_event": done,
        "response_characters": len(text),
        "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "content_piece_span_seconds": round(decode_span, 3) if decode_span is not None else None,
        "content_pieces_per_second": round((len(pieces) - 1) / decode_span, 2)
        if decode_span and decode_span > 0 else None,
        "completion_tokens_per_http_second": round(tokens / wall, 2)
        if isinstance(tokens, int) and wall > 0 else None,
    }
    result["ok"] = (not errors and done and finish_reason in {"stop", "length"}
                    and isinstance(tokens, int) and tokens >= 160
                    and len(pieces) >= 120 and decode_span is not None
                    and "<unused" not in text)
    if errors:
        result["errors"] = errors
    return result


def embed_batch(base: str, alias: str, trial: int) -> dict:
    started = time.perf_counter_ns()
    payload = post_json(base + "/v1/embeddings", {"model": alias, "input": EMBED_INPUTS})
    wall = (time.perf_counter_ns() - started) / 1e9
    vectors = [item.get("embedding") for item in payload.get("data", [])]
    result = {
        "trial": trial,
        "http_wall_seconds": round(wall, 3),
        "vectors": len(vectors),
        "dimensions": sorted({len(vector) for vector in vectors if isinstance(vector, list)}),
        "vectors_per_http_second": round(len(vectors) / wall, 2) if wall > 0 else None,
    }
    result["ok"] = len(vectors) == len(EMBED_INPUTS) and result["dimensions"] == [768]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--dll-dir", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kind", choices=("chat", "embedding"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eie-commit", required=True)
    parser.add_argument("--prismml-commit", required=True)
    args = parser.parse_args()
    for path in (args.exe, args.preset, args.model):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    if not args.dll_dir.is_dir():
        parser.error("missing DLL directory")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".log")
    alias = args.model.stem
    port = free_port()
    command = [str(args.exe.resolve()), "--config", str(args.preset.resolve()),
               "--model", str(args.model.resolve()), "--host", "127.0.0.1",
               "--port", str(port)]
    env = dict(os.environ)
    paths = [str(args.dll_dir.resolve()), str(args.exe.resolve().parent)]
    if env.get("HIP_PATH"):
        paths.append(str(Path(env["HIP_PATH"]) / "bin"))
    env["PATH"] = os.pathsep.join([*paths, env.get("PATH", "")])
    receipt = {
        "schema": "eie.windows-hip-x2-throughput/v1",
        "scope": "isolated EIE process, one GGUF, synthetic input, loopback, three measured trials",
        "utc": datetime.now(timezone.utc).isoformat(),
        "eie_commit": args.eie_commit,
        "prismml_commit": args.prismml_commit,
        "exe_filename": args.exe.name,
        "exe_sha256": sha256(args.exe),
        "preset_filename": args.preset.name,
        "preset_sha256": sha256(args.preset),
        "model_filename": args.model.name,
        "model_sha256": sha256(args.model),
        "kind": args.kind,
        "gpu_expected": "ROCm0 gfx1151",
        "prompt": CHAT_PROMPT if args.kind == "chat" else None,
        "embedding_input_count": len(EMBED_INPUTS) if args.kind == "embedding" else None,
        "trials": [],
        "ok": False,
    }
    process = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command, cwd=str(args.dll_dir.resolve()), env=env,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 300
            started = time.monotonic()
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"EIE exited during load: {process.returncode}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=2) as response:
                        health = json.load(response)
                    if health.get("models") == 1:
                        break
                    if health.get("models") == 0:
                        raise RuntimeError("EIE started without a model")
                except (OSError, urllib.error.HTTPError, ValueError):
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError("EIE did not report one loaded model")
            receipt["startup_seconds"] = round(time.monotonic() - started, 3)
            if args.kind == "chat":
                warmup = post_json(base + "/v1/chat/completions", {
                    "model": alias, "messages": [{"role": "user", "content": "Bonjour."}],
                    "max_tokens": 32, "temperature": 0, "stream": False})
                receipt["warmup_completion_tokens"] = warmup["usage"]["completion_tokens"]
            else:
                warmup = post_json(base + "/v1/embeddings", {
                    "model": alias, "input": "Document synthetique de controle."})
                receipt["warmup_dimensions"] = len(warmup["data"][0]["embedding"])
            for trial in range(1, 4):
                row = stream_chat(base, alias, trial) if args.kind == "chat" else embed_batch(base, alias, trial)
                receipt["trials"].append(row)
                print(json.dumps({"trial": trial, "ok": row["ok"],
                                  "rate": row.get("content_pieces_per_second", row.get("vectors_per_http_second"))}),
                      flush=True)
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
        if log_path.is_file():
            receipt["private_log_sha256"] = sha256(log_path)
            text = log_path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"offloaded (\d+)/(\d+) layers to GPU", text)
            receipt["gpu_layers_offloaded"] = [int(match.group(1)), int(match.group(2))] if match else None
            receipt["rocm_device_in_log"] = "ROCm0" in text
        offload = receipt.get("gpu_layers_offloaded")
        receipt["ok"] = ("error" not in receipt and len(receipt["trials"]) == 3
                         and all(row["ok"] for row in receipt["trials"])
                         and bool(offload) and offload[0] == offload[1]
                         and receipt.get("rocm_device_in_log") is True)
        if receipt["ok"]:
            key = "content_pieces_per_second" if args.kind == "chat" else "vectors_per_http_second"
            receipt["median_rate"] = round(statistics.median(row[key] for row in receipt["trials"]), 2)
        args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": receipt["ok"], "out": str(args.out), "error": receipt.get("error")}), flush=True)
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
