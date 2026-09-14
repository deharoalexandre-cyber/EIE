"""Publish checked Android receipts and byte-identical, test-scoped raw outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=3, action="append", required=True,
                        metavar=("LABEL", "RUN_DIR", "BUNDLE_DIR"))
    parser.add_argument("--out", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model-bytes", type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    rows = []
    for label, run_dir, bundle_dir in args.run:
        if not label.replace("-", "").isalnum():
            raise ValueError("Use a simple alphanumeric run label")
        source, bundle = Path(run_dir), Path(bundle_dir)
        report_path = source / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        assert sha(bundle / "manifest.json") == report["bundle_manifest_sha256"]
        for name, digest in report["library_sha256"].items():
            assert manifest["files"][name]["sha256"] == digest
        target = args.out / label
        target.mkdir()
        excluded = []
        published = {}
        for name, info in report["artifacts"].items():
            if Path(name).name != name:
                raise ValueError("Artifact must be a basename")
            raw = source / name
            assert sha(raw) == info["sha256"] and raw.stat().st_size == info["bytes"], name
            # Early probes captured Samsung charging history. Keep that private;
            # later validators request only current level, temperature and USB.
            if name.startswith(("battery-before.", "battery-after.")):
                excluded.append({"name": name, "reason": "charging history outside the test scope"})
                continue
            shutil.copy2(raw, target / name)
            published[name] = info
        report["artifacts"] = published
        report["publication"] = {
            "original_report_sha256": sha(report_path), "excluded_artifacts": excluded,
            "raw_outputs": "Copied byte-for-byte; report artifact index is filtered only for privacy",
        }
        write_json(target / "report.json", report)
        shutil.copy2(bundle / "manifest.json", target / "bundle-manifest.json")
        shutil.copy2(bundle / "validate_android_neon.py", target / "validator.py")
        shutil.copytree(bundle / "evidence", target / "build-evidence")
        for request in report["requests"]:
            rows.append({"run": label, "variant": report["variant"], "cpu_arch": report["cpu_arch"],
                         "device": report["device"], "binary_sha256": report["binary_sha256"],
                         "library_sha256": report["library_sha256"], "model": report["model"],
                         "source_revision": report["source_revision"], "settings": report["settings"], **request})
        results.append({"label": label, "pass": report["pass"], "source_revision": report["source_revision"],
                        "variant": report["variant"], "manifest_sha256": report["bundle_manifest_sha256"],
                        "unit_stdout": report["unit_stdout"], "excluded_artifacts": excluded})
    summary = {"schema": "eie.android.public-campaign/v1", "runs": results, "measurements": rows,
               "rate_definition": "API completion_tokens / complete host request wall time, including prefill and transport; not decode-only",
               "model_bytes": args.model_bytes,
               "scope": "Maintainer-run CPU text-chat validation on one Z Flip6 / SM8650 / Android 16, not independent replication",
               "limitations": ["Two short prompts; 64-token length-limited generations are intentionally incomplete",
                               "Sequential, non-randomized trials; uncontrolled background load and cache/thermal state",
                               "No matched CPU/GPU/NPU comparison, Android EWS, vision, JNI or APK qualification"]}
    write_json(args.out / "summary.json", summary)
    inventory = {p.relative_to(args.out).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                 for p in sorted(args.out.rglob("*")) if p.is_file()}
    write_json(args.out / "inventory.json", inventory)
    archive = args.out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(args.out.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(args.out).as_posix())
    summary["evidence_archive"] = {"name": archive.name, "sha256": sha(archive), "bytes": archive.stat().st_size}
    write_json(args.summary, summary)
    print(json.dumps(summary["evidence_archive"]))


if __name__ == "__main__":
    main()
