"""Run matched forwards against EIE's pinned runtime, without Next resident.

Outputs are private build artifacts, never overwritten. No model download.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--predict", type=int, default=32)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    build = ROOT / "build-ews"
    binary = build / "Release/ews-forward.exe"
    env = dict(os.environ)
    env["PATH"] = str(build / "bin/Release") + os.pathsep + env.get("PATH", "")
    # Both arms use the same explicit CUDA graph profile as the original pilot.
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    report = {"runtime_pin": "2168b0cd8b87c75c29a1e6588692ebbb805b9bd2",
              "cuda_graphs": False, "comparisons": []}
    for domain in ("fr", "code"):
        prompt = ROOT / f"experiments/ews/prompts/{domain}_gemma4.txt"
        if not prompt.exists():
            raise FileNotFoundError(prompt)
        for slots in (0, 32, 8):
            prefix = args.out / f"{domain}-{slots}"
            with prefix.with_suffix(".log").open("wb") as log:
                subprocess.run([str(binary), args.model, str(prompt), str(prefix), str(slots), str(args.predict)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
            print(f"completed {domain}, slots={slots}", flush=True)
        ref = np.fromfile(args.out / f"{domain}-0.logits.bin", dtype=np.float32)
        ref_tokens = json.loads((args.out / f"{domain}-0.json").read_text())["generated_token_ids"]
        for slots in (32, 8):
            path = args.out / f"{domain}-{slots}.logits.bin"
            candidate = np.fromfile(path, dtype=np.float32)
            same = ref.shape == candidate.shape and np.array_equal(ref.view(np.uint32), candidate.view(np.uint32))
            item = {"domain": domain, "slots": slots, "floats": int(ref.size), "bit_identical": same,
                    "same_generated_tokens": ref_tokens == json.loads((args.out / f"{domain}-{slots}.json").read_text())["generated_token_ids"],
                    "max_abs_delta": float(np.max(np.abs(ref - candidate))) if ref.shape == candidate.shape else None,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            report["comparisons"].append(item)
            print(json.dumps(item), flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all(x["bit_identical"] for x in report["comparisons"]):
        raise SystemExit("Numerical gate failed; do not run Next envelope yet")

if __name__ == "__main__":
    main()
