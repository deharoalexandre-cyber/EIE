#!/usr/bin/env python3
"""Publish a path-free summary of three private EIE Windows HIP receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = {}
    binary_hashes = set()
    for name, expected_kind in (("gemma", "chat"), ("bonsai", "chat"),
                                ("nomic", "embedding")):
        receipt_path = args.input_dir / f"eie-hip-{name}.json"
        log_path = args.input_dir / f"eie-hip-{name}.log"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["schema"] == "eie.windows-hip-isolated/v1"
        assert receipt["kind"] == expected_kind
        assert receipt["ok"] is True
        assert receipt["expected_backend"] == "rocm"
        assert receipt["rocm_device_in_log"] is True
        assert receipt["health"]["models"] == 1
        layers = receipt["gpu_layers_offloaded"]
        assert layers[0] == layers[1] and layers[0] > 0
        trials = receipt["trials"]
        assert len(trials) == 2 and all(t["ok"] for t in trials)
        if expected_kind == "embedding":
            assert all(t["dimensions"] == 768 for t in trials)
        else:
            assert all(t["finish_reason"] == "stop" and t["content_excerpt"].strip()
                       for t in trials)
        binary_hashes.add(receipt["exe_sha256"])
        cases[name] = {
            "model": receipt["model_alias"],
            "model_sha256": receipt["model_sha256"],
            "kind": expected_kind,
            "passed": True,
            "timestamp": receipt["utc"],
            "startup_seconds": receipt["startup_seconds"],
            "health_models": 1,
            "rocm_device_in_log": True,
            "gpu_layers_offloaded": layers,
            "trials": [({"ok": True, "wall_seconds": t["wall_seconds"],
                        "dimensions": t["dimensions"]} if expected_kind == "embedding"
                       else {"ok": True, "wall_seconds": t["wall_seconds"],
                             "completion_tokens": t["usage"]["completion_tokens"],
                             "finish_reason": t["finish_reason"]}) for t in trials],
            "private_raw_receipt_sha256": sha256(receipt_path),
            "private_raw_log_sha256": sha256(log_path),
        }
    assert len(binary_hashes) == 1, "cases used different EIE executables"
    summary = {
        "schema": "eie.windows-hip-x2-summary/v1",
        "scope": "three separate one-model EIE runs, loopback, synthetic prompts",
        "platform": "GMKtec EVO-X2, Windows 11 build 26100, AMD Radeon 8060S gfx1151",
        "toolchain": "HIP SDK 7.2, Visual Studio Build Tools 2022, clang++, Ninja",
        "eie_source_commit": "527c4501415ce871b1f66bf2175e50d53fe0d17d",
        "prismml_commit": "a15369706017f177cae8aa40b6b8592259a48806",
        "eie_executable_sha256": next(iter(binary_hashes)),
        "build_profile": {
            "EIE_LLAMA_RUNTIME": "prism", "GGML_HIP": "ON",
            "GGML_BACKEND_DL": "ON", "GPU_TARGETS": "gfx1151",
        },
        "cases": cases,
        "limits": [
            "one machine and one run per model; no statistical latency estimate",
            "models loaded separately; no simultaneous multi-model run",
            "no EWS, long-context, sustained-load, or full client application validation",
            "request wall times include transport and prompt processing, not isolated decode",
            "private raw receipts and logs retained on the qualification machine",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
