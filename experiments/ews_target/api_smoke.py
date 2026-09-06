"""Real EIE HTTP regression checks: streaming, one-shot, cache accounting, KV names."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import urllib.error
from next_envelope import http, ROOT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    reports = []
    env = dict(os.environ)
    env["PATH"] = str(ROOT / "build-ews/bin/Release") + os.pathsep + env.get("PATH", "")
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    base = "http://127.0.0.1:18080"
    for kv in ("f16", "turbo3"):
        cfg = args.out / (kv + ".yaml")
        cfg.write_text(f"host: 127.0.0.1\nport: 18080\nauto_discover: false\ntype_k: {kv}\ntype_v: {kv}\nn_ctx: 512\nflash_attn: true\nmodels:\n  ews: {args.model}\news_slots:\n  ews: 16\npreload: [ews]\n")
        with (args.out / (kv + ".log")).open("wb") as log:
            process = subprocess.Popen([str(ROOT / "build-ews/Release/eie-server.exe"), "--config", str(cfg)], env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(120):
                    if process.poll() is not None: raise RuntimeError("EIE exited")
                    try:
                        catalog = http(base + "/v1/models", timeout=1)
                        break
                    except OSError: time.sleep(1)
                else: raise TimeoutError("EIE startup")
                assert catalog["data"] and catalog["data"][0]["id"] == "ews", catalog
                assert set(catalog["data"][0]) == {"id", "object", "owned_by"}, catalog
                prompt = {"model": "ews", "temperature": 0, "max_tokens": 24,
                          "messages": [{"role": "user", "content": "En trois phrases, explique un cache LRU."}]}
                outputs = []
                for oneshot in (False, True, False):
                    response = http(base + "/v1/chat/completions", dict(prompt, one_shot=oneshot))
                    assert response["choices"][0]["message"]["content"]
                    outputs.append(response["choices"][0]["message"]["content"])
                # One-shot must not contaminate the persistent chat's computation.
                assert len(set(outputs)) == 1, "One-shot / repeated chat diverged"
                request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(dict(prompt, stream=True)).encode(),
                                                  headers={"Content-Type": "application/json"})
                chunks = []
                with urllib.request.urlopen(request, timeout=120) as response:
                    events = response.read().decode()
                assert "data: [DONE]" in events and '"error"' not in events, events
                for line in events.splitlines():
                    if line.startswith("data: {"):
                        chunks.append(json.loads(line[6:]))
                text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
                assert text == outputs[0], "SSE / non-SSE content diverged"
                stats = http(base + "/v1/admin/ews/status")["ews"]
                assert stats["callbacks"] > 0 and stats["misses"] > 0
                assert stats["hits"] + stats["misses"] == 8 * stats["callbacks"]
                assert stats["physical_expert_bytes"] * 8 == stats["logical_expert_bytes"]
                reports.append({"kv": kv, "chat_oneshot_repeated_identical": True, "sse_identical": True,
                                "catalog_compatible": True, "stats": stats})
                print(json.dumps(reports[-1]), flush=True)
            finally:
                if process.poll() is None: process.terminate()
                process.wait(timeout=30)
        contents = (args.out / (kv + ".log")).read_text(errors="replace")
        assert f"kv={kv}/{kv}" in contents and "retrying f16" not in contents, "KV fallback occurred"
    (args.out / "report.json").write_text(json.dumps(reports, indent=2) + "\n")

if __name__ == "__main__": main()
