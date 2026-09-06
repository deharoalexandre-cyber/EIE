"""Offline evidence inventory. Hash agreement is not an inference rerun.

Reads Git blobs so Windows checkout line endings cannot silently change the
comparison. Historical manifests were made from local files; report CRLF
matches separately from byte-exact Git-blob matches. Never edits the archive.
"""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "07b1500e6b5f68b5fb12a3118aff832d217bbfc6"
PREFIX = "docs/benchmarks/data/ews/"
EXPECTED = {
    "ews_p1a2_engine_frozen.cpp": "45db589b636414c7b808f10b07ba99835448bf02b982624332f16ee4220cae62",
    "replay.py": "ae039126eaf9bbb9ef1774ed5261cea3ff64642fa02ee925b4bb35126dd565af",
    "verdict_cplus.py": "2d424557c27364371d55d325f0ec5eab50c5df0be3d2d65f4e56e5248bfd9704",
    "VERDICT-CPLUS-PROTOCOL.md": "1d530b8d84447820f93223b188d5718142d44cfd5858dbf8fa5d63d805f2efc5",
    "AMENDMENT-CPLUS-v1.1.md": "6727ad548cedda836d95671f1de8001b1f79ee89f6fc4bbdd1741112ba0d148f",
    "KILL-THRESHOLD-v3.md": "7ee481219f42c720d419ec20fdadaefcb2d4a38fe4198d4a78e6621148d39254",
    "P1-GATE.md": "cc64e4e1484f54182a2eb5bd3c8eb5f07c846ca0844e99949103b2d874f8eb43",
}


def blob(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)


def main():
    records = []
    for name, expected in EXPECTED.items():
        raw = blob(PREFIX + name)
        lf = raw.replace(b"\r\n", b"\n")
        candidates = {"git_blob": raw, "lf": lf, "crlf": lf.replace(b"\n", b"\r\n")}
        hashes = {label: hashlib.sha256(data).hexdigest() for label, data in candidates.items()}
        matches = [label for label, digest in hashes.items() if digest == expected]
        records.append({"file": PREFIX + name, "expected": expected, "hashes": hashes, "matches": matches})
    manifest = blob(PREFIX + "HASHES.txt").decode("utf-8")
    import re
    prompts = sorted(set(re.findall(r"\b(g\d+p?_[a-z0-9_]+\.txt)\b", manifest)))
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE], cwd=ROOT, text=True).splitlines()
    missing = [name for name in prompts if not any(p.endswith("/" + name) for p in tracked)]
    summary = json.loads(blob("docs/benchmarks/data/gemma4-26b-a4b-rtx4090-laptop.json"))
    gains = {name: round(100 * (1 - engine / static), 3) for name, engine, static in [
        ("G5p", 159.7, 266.9), ("G6", 97.1, 181.8), ("G7", 204.1, 396.6), ("G8p", 139.0, 245.3)]}
    print(json.dumps({
        "audited_revision": BASE,
        "not_an_inference_rerun": True,
        "hash_checks": records,
        "missing_manifest_prompt_files": missing,
        "ews_reduction_percent_from_reported_rounded_values": gains,
        "legacy_26b_gpu_used_gib": [round(x / 1024, 3) for x in summary["memory"]["gpu_used_mib_range"]],
        "evidence_limit": "The original prompts, routing traces, raw timing runs, tracer source and benchmark build recipe are not a complete public reproduction bundle."
    }, indent=2))


if __name__ == "__main__":
    main()
