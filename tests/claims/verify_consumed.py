"""Check September EWS summaries and, optionally, existing local raw evidence.

No inference or GPU access. A summary-only check is not raw verification.
"""
import argparse
import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "docs/benchmarks/data"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(root, name, expected, size=None):
    root = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evidence path outside supplied directory")
    assert path.is_file(), f"Missing evidence: {name}"
    if size is not None:
        assert path.stat().st_size == size, f"Size mismatch: {name}"
    assert digest(path) == expected, f"Digest mismatch: {name}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-raw", type=Path)
    ap.add_argument("--runtime-raw", type=Path)
    ap.add_argument("--next-build-root", type=Path)
    args = ap.parse_args()
    pilot = json.loads((DATA / "ews-consumed-pilot-20260905.json").read_text())
    target = json.loads((DATA / "ews-runtime-integration-20260905.json").read_text())
    positives = negatives = files = 0
    for case in pilot["cases"]:
        assert case["passed"] and case["exit_code"] == 0
        if case["slots_per_layer"]:
            assert case["physical_expert_bytes"] == 12846366720 * case["slots_per_layer"] // 128
            for phase in ("prefill", "decode"):
                stats = case[phase]
                assert stats["hits"] + stats["misses"] == stats["accesses"]
                assert stats["direct_read_bytes"] >= stats["payload_h2d_bytes"] > 0
            comparison = case["comparison"]
            if comparison["negative_control"]:
                assert not comparison["logits_byte_identical"] and comparison["max_abs_logit_difference"] > 0
                negatives += 1
            else:
                assert comparison["logits_byte_identical"] and comparison["max_abs_logit_difference"] == 0
                assert comparison["reference_logits_sha256"] == comparison["candidate_logits_sha256"]
                positives += 1
        if args.pilot_raw:
            for item in case["evidence"]:
                verify_file(args.pilot_raw, item["file"], item["sha256"], item["bytes"])
                files += 1
            raw = json.loads((args.pilot_raw / f"{case['case']}.json").read_text())
            for key in ("physical_expert_bytes", "prefill", "decode"):
                assert raw[key] == case[key], f"Summary differs from raw {case['case']} {key}"
    assert positives == 5 and negatives == 1
    checked_runtime = set()
    for item in target["numerical_gate"]["comparisons"]:
        assert item["bit_identical"] and item["same_generated_tokens"] and item["max_abs_delta"] == 0
        if args.runtime_raw:
            for slots in (0, item["slots"]):
                name = f"{item['domain']}-{slots}.logits.bin"
                if name not in checked_runtime:
                    verify_file(args.runtime_raw, name, item["sha256"], item["floats"] * 4)
                    checked_runtime.add(name)
    for run in target["envelopes"]:
        if args.next_build_root:
            verify_file(args.next_build_root, run["run"] + "/report.json", run["raw_report_sha256"])
    print(json.dumps({
        "inference_rerun": False,
        "pilot_positive_comparisons": positives,
        "pilot_negative_control": negatives,
        "pilot_raw_files_sha256_and_size_verified": files,
        "runtime_raw_logits_files_sha256_and_size_verified": len(checked_runtime),
        "next_raw_reports_sha256_verified": len(target["envelopes"]) if args.next_build_root else 0,
        "pilot_raw_status": "verified" if args.pilot_raw else "not supplied",
        "runtime_raw_status": "verified" if args.runtime_raw else "not supplied",
        "next_raw_status": "hashes verified, contents not disclosed" if args.next_build_root else "not supplied",
        "limitation": "Checks archived evidence, not a fresh performance, quality or deployment campaign."
    }, indent=2))


if __name__ == "__main__":
    main()
