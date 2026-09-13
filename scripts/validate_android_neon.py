"""Capture a raw, isolated Android EIE CPU campaign (no app data collection)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", required=True, help="Used for transport, not included in public metadata")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Absolute device path readable by the ADB shell")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for relative, info in manifest["files"].items():
        if sha(bundle / relative) != info["sha256"]:
            raise SystemExit("Bundle hash mismatch: " + relative)
    args.out.mkdir(parents=True, exist_ok=False)
    adb_base = [args.adb, "-s", args.serial]

    def adb(parts, label, check=True, timeout=120):
        result = subprocess.run(adb_base + parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        (args.out / (label + ".stdout")).write_bytes(result.stdout)
        (args.out / (label + ".stderr")).write_bytes(result.stderr)
        if check and result.returncode:
            raise RuntimeError(f"{label}: exit {result.returncode}: {result.stderr.decode(errors='replace')}")
        return result.stdout.decode("utf-8", errors="replace").strip()

    def shell(command, label, **kwargs):
        return adb(["shell", command], label, **kwargs)

    device = {}
    for prop in ("ro.product.model", "ro.product.manufacturer", "ro.soc.model", "ro.board.platform",
                 "ro.product.cpu.abilist", "ro.build.version.release", "ro.build.version.sdk", "ro.build.version.security_patch"):
        device[prop] = shell("getprop " + prop, "device-" + prop)
    shell("cat /proc/cpuinfo", "device-cpuinfo")
    shell("cat /proc/meminfo", "memory-before")
    for field in ("level", "temp", "usb"):
        shell("dumpsys battery get " + field, "battery-before-" + field)
    shell("dumpsys thermalservice", "thermal-before", check=False)
    model_hash = shell("sha256sum " + shlex.quote(args.model), "model-hash", timeout=240).split()[0]
    if not re.fullmatch(r"[0-9a-f]{64}", model_hash):
        raise RuntimeError("No valid model SHA-256")
    remote = "/data/local/tmp/eie-neon-" + manifest["variant"] + "-" + str(time.time_ns())
    shell("mkdir " + shlex.quote(remote), "mkdir")
    adb(["push", str(bundle / "bin"), remote + "/bin"], "push-bin")
    adb(["push", str(bundle / "lib"), remote + "/lib"], "push-lib")
    remote_hashes = {}
    for relative in sorted(manifest["files"]):
        if not relative.startswith(("bin/", "lib/")):
            continue
        digest = shell("sha256sum " + shlex.quote(remote + "/" + relative), "hash-" + relative.replace("/", "-"))
        remote_hashes[relative] = digest.split()[0]
        if remote_hashes[relative] != manifest["files"][relative]["sha256"]:
            raise RuntimeError("Device artifact mismatch: " + relative)
    shell("chmod 755 " + shlex.quote(remote + "/bin/eie-server") + " " + shlex.quote(remote + "/bin/eie-serving-tests"), "chmod")
    env = "LD_LIBRARY_PATH=" + shlex.quote(remote + "/lib")
    unit = shell(env + " " + shlex.quote(remote + "/bin/eie-serving-tests"), "serving-unit")
    device_port = 18761
    config = ("host: 127.0.0.1\nport: 18761\nauto_discover: false\ntype_k: f16\ntype_v: f16\n"
              "flash_attn: true\nn_ctx: 1024\nmodels:\n  resident: " + args.model +
              "\nthreads:\n  resident: 4\npreload: [resident]\n")
    config_path = args.out / "eie.yaml"
    config_path.write_text(config, encoding="utf-8")
    adb(["push", str(config_path), remote + "/eie.yaml"], "push-config")
    host_port = int(adb(["forward", "tcp:0", f"tcp:{device_port}"], "forward"))
    report = {"schema": "eie.android.device/v1", "pass": False, "device": device,
              "started_utc": datetime.now(timezone.utc).isoformat(), "validator_sha256": sha(Path(__file__)),
              "variant": manifest["variant"], "cpu_arch": manifest["cpu_arch"],
              "binary_sha256": remote_hashes["bin/eie-server"], "library_sha256": remote_hashes,
              "bundle_manifest_sha256": sha(bundle / "manifest.json"),
              "model": {"name": args.model_name, "quantization": args.quantization, "sha256": model_hash},
              "settings": {"context": 1024, "type_k": "f16", "type_v": "f16", "threads": 4,
                           "temperature": 0, "one_shot": True, "openmp": False},
              "source_revision": manifest["source_revision"], "runtime_revision": manifest["runtime_revision"],
              "raw_logs": True, "requests": [], "unit_stdout": unit}
    log_path = args.out / "server-adb.raw.log"
    command = "echo EIE_TEST_PID=$$; exec env " + env + " " + shlex.quote(remote + "/bin/eie-server") + " --config " + shlex.quote(remote + "/eie.yaml")
    pid = None
    with log_path.open("wb") as log:
        process = subprocess.Popen(adb_base + ["shell", command], stdout=log, stderr=subprocess.STDOUT)

        def request(path, body=None, label="request"):
            encoded = None if body is None else json.dumps(body).encode()
            if encoded:
                (args.out / (label + ".request.json")).write_bytes(encoded)
            req = urllib.request.Request(f"http://127.0.0.1:{host_port}" + path, data=encoded,
                                         headers={"Content-Type": "application/json"})
            started = time.monotonic()
            first = None
            raw = bytearray()
            chunks = []
            with urllib.request.urlopen(req, timeout=240) as response:
                if body and body.get("stream"):
                    for line in response:
                        raw.extend(line)
                        if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                            event = json.loads(line[6:])
                            chunks.append(event)
                            if first is None and event["choices"][0]["delta"].get("content"):
                                first = time.monotonic() - started
                else:
                    raw.extend(response.read())
            wall = time.monotonic() - started
            (args.out / (label + ".response.raw")).write_bytes(raw)
            if chunks:
                if not raw.rstrip().endswith(b"data: [DONE]"):
                    raise RuntimeError("Missing SSE terminator")
                text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
                usage, finish = chunks[-1]["usage"], chunks[-1]["choices"][0]["finish_reason"]
            else:
                value = json.loads(raw)
                if body is None:
                    return value
                text, finish, usage = value["choices"][0]["message"]["content"], value["choices"][0]["finish_reason"], value["usage"]
            result = {"label": label, "text": text, "finish_reason": finish, "usage": usage,
                      "request_wall_seconds": wall, "first_content_seconds": first,
                      "output_tokens_per_request_wall_second": usage["completion_tokens"] / wall}
            report["requests"].append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "text"}), flush=True)
            return result

        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                match = re.search(rb"EIE_TEST_PID=(\d+)", log_path.read_bytes())
                if match:
                    pid = match[1].decode()
                if process.poll() is not None:
                    raise RuntimeError("Server exited during startup")
                try:
                    catalog = request("/v1/models", label="models")
                    if {m["id"] for m in catalog["data"]} == {"resident"}:
                        break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Model startup")
            body = {"model": "resident", "messages": [{"role": "user", "content": "What is 17 + 25? Reply with only the integer."}],
                    "temperature": 0, "max_tokens": 32, "one_shot": True, "truncate_prompt": False, "strict_model": True}
            normal = request("/v1/chat/completions", body, "arithmetic-buffered")
            body["stream"] = True
            streamed = request("/v1/chat/completions", body, "arithmetic-streamed")
            assert normal["text"].strip() == "42", normal
            assert normal["text"] == streamed["text"] and normal["finish_reason"] == streamed["finish_reason"] == "stop"
            for i in range(args.repeats):
                body["messages"][0]["content"] = "Describe how a bicycle works in a clear paragraph of about one hundred words."
                body["max_tokens"] = 64
                result = request("/v1/chat/completions", body, f"generation-{i+1}")
                assert result["text"] and 0 < result["usage"]["completion_tokens"] <= 64
                assert result["finish_reason"] in ("stop", "length")
            report["pass"] = True
        except BaseException as exc:
            report["error"] = str(exc)
            raise
        finally:
            if pid:
                cmdline = shell("cat /proc/" + pid + "/cmdline", "owned-process", check=False)
                if remote + "/bin/eie-server" in cmdline:
                    shell("kill -TERM " + pid, "stop-owned-server", check=False)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
            adb(["forward", "--remove", "tcp:" + str(host_port)], "remove-forward", check=False)
            shell("cat /proc/meminfo", "memory-after")
            for field in ("level", "temp", "usb"):
                shell("dumpsys battery get " + field, "battery-after-" + field)
            shell("dumpsys thermalservice", "thermal-after", check=False)
            report["artifacts"] = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                                   for p in sorted(args.out.iterdir()) if p.is_file()}
            (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Android campaign passed; raw logs retained in " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
