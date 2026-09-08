"""Run matched forwards against EIE's pinned runtime, without Next resident.

Outputs are private build artifacts, never overwritten. No model download.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--predict", type=int, default=32)
    ap.add_argument("--profile", choices=("gemma-gpu", "glm-cpu"), default="gemma-gpu")
    ap.add_argument("--slots", type=int, nargs="+", default=[32, 8])
    ap.add_argument("--build", type=Path, default=ROOT / "build-ews")
    ap.add_argument("--dll-dir", type=Path)
    ap.add_argument("--runtime-source", type=Path, default=ROOT / "llama.cpp")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    build = args.build.resolve()
    dll_dir = (args.dll_dir or build / "bin/Release").resolve()
    binary = build / "Release/ews-forward.exe"
    env = dict(os.environ)
    env["PATH"] = str(dll_dir) + os.pathsep + env.get("PATH", "")
    # Both arms use the same explicit CUDA graph profile as the original pilot.
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    report = {"runtime_pin": subprocess.check_output(["git", "-C", str(args.runtime_source), "rev-parse", "HEAD"], text=True).strip(),
              "profile": args.profile, "cuda_graphs": False, "tf32": False, "comparisons": [], "runs": []}
    report["binary_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (binary, dll_dir / "llama.dll")}
    report["runtime_diff_sha256"] = hashlib.sha256(subprocess.check_output(["git", "-C", str(args.runtime_source), "diff"])).hexdigest()
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    for domain in (("glm-smoke",) if args.profile == "glm-cpu" else ("fr", "code")):
        prompt = (ROOT / "experiments/ews_target/glm53-smoke-prompt.txt" if args.profile == "glm-cpu"
                  else ROOT / f"experiments/ews/prompts/{domain}_gemma4.txt")
        if not prompt.exists():
            raise FileNotFoundError(prompt)
        for slots in (0, *args.slots):
            prefix = args.out / f"{domain}-{slots}"
            command = [str(binary), args.model, str(prompt), str(prefix), str(slots), str(args.predict)]
            if args.profile == "glm-cpu": command.append(args.profile)
            started = time.monotonic()
            print(f"starting {domain}, slots={slots}", flush=True)
            with prefix.with_suffix(".log").open("wb") as log:
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                while True:
                    try:
                        code = proc.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        print(f"running {domain}, slots={slots}, pid={proc.pid}, elapsed={time.monotonic()-started:.0f}s", flush=True)
                report["runs"].append({"domain": domain, "slots": slots, "exit_code": code,
                                       "seconds": time.monotonic()-started})
                (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                if code != 0: raise SystemExit(f"forward failed ({code}); see {prefix}.log")
            print(f"completed {domain}, slots={slots}", flush=True)
        ref = np.fromfile(args.out / f"{domain}-0.logits.bin", dtype=np.float32)
        ref_tokens = json.loads((args.out / f"{domain}-0.json").read_text())["generated_token_ids"]
        for slots in args.slots:
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
