#!/usr/bin/env python3
# EIE: Windows validation receipt. Port of scripts/receipt-macos.sh: starts the
# engine of a bundle on a test port, measures health / embeddings / chat with a
# cold then warm prefix, and writes a JSON in the repository receipt format
# (docs/benchmarks/data/). Standard library only.
# Usage: python scripts\receipt-windows.py --bundle DIR --models DIR [--port 8099]
#        [--out FILE.json] [--rev GITREV] [--preset windows-cpu.yaml]
# NEVER run it against an engine in service: it starts its own instance.
import argparse, ctypes, glob, hashlib, json, os, platform, re, subprocess, sys, tempfile, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--port", type=int, default=8099)
ap.add_argument("--out", default="")
ap.add_argument("--rev", default="unknown")
ap.add_argument("--preset", default="windows-cpu.yaml")
a = ap.parse_args()

exe = os.path.join(a.bundle, "eie-server.exe")
if not os.path.isfile(exe): sys.exit("--bundle DIR must contain eie-server.exe")
if not os.path.isdir(a.models): sys.exit("--models DIR must contain the GGUF files")
ggufs = sorted(glob.glob(os.path.join(a.models, "*.gguf")))
if len(ggufs) < 2: sys.exit("need one generation model and one embedding model (name containing 'bge' or 'embed')")
arch = platform.machine().lower().replace("amd64", "x64")
out = a.out or f"receipt-windows-{arch}-{time.strftime('%Y%m%d')}.json"

work = tempfile.mkdtemp(prefix="eie-receipt-")
ptxt = open(os.path.join(a.bundle, "presets", a.preset), encoding="utf-8").read()
ptxt = re.sub(r"^port:.*$", f"port: {a.port}", ptxt, flags=re.M)
ptxt = re.sub(r"^model_dir:.*$", "model_dir: " + os.path.abspath(a.models).replace("\\", "/"), ptxt, flags=re.M)
preset_path = os.path.join(work, "preset.yaml")
open(preset_path, "w", encoding="utf-8").write(ptxt)
log_path = os.path.join(work, "engine.log")
log_f = open(log_path, "w", encoding="utf-8", errors="replace")
t0 = time.time()
proc = subprocess.Popen([exe, "--config", preset_path], stdout=log_f, stderr=subprocess.STDOUT, cwd=a.bundle)
base = f"http://127.0.0.1:{a.port}"


def get(path, timeout):
    return urllib.request.urlopen(base + path, timeout=timeout).read().decode()


try:
    health = None
    for _ in range(600):
        if proc.poll() is not None: sys.exit("engine exited early, see " + log_path)
        try:
            h = json.loads(get("/health", 2))
            if h.get("models") == len(ggufs): health = h; break
        except Exception: pass
        time.sleep(1)
    if health is None: sys.exit("engine not healthy after 600 s, see " + log_path)
    load_s = round(time.time() - t0, 2)
    print(f"engine ready in {load_s}s: {json.dumps(health)}")

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=1800)); return r, round(time.time() - t, 3)

    def kv_lines():
        log_f.flush()
        return [l for l in open(log_path, encoding="utf-8", errors="replace") if l.startswith("[KV] reused=")]

    def parse_kv(line):
        return {k: int(v) for k, v in re.findall(r"(reused|gen|total_ms)=(-?\d+)", line)}

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
        return h.hexdigest()

    def stem(p): return os.path.basename(p)[:-5]
    def is_emb(p): return "bge" in os.path.basename(p).lower() or "embed" in os.path.basename(p).lower()
    gen_model = next(stem(p) for p in ggufs if not is_emb(p))
    emb_model = next(stem(p) for p in ggufs if is_emb(p))

    # 1. embeddings
    emb, emb_s = post("/v1/embeddings", {"model": emb_model, "input": "Elyne vit localement dans ce PC."})
    vec = emb["data"][0]["embedding"]; norm = sum(x * x for x in vec) ** 0.5

    # 2. chat, cold prefix (fixed context of about 250 tokens)
    system = ("Tu es un assistant de test. " * 20).strip()
    history = [{"role": "system", "content": system},
               {"role": "user", "content": "Explique en trois phrases ce qu'est un cache KV dans un modèle de langage."}]

    def chat(max_tokens):
        n = len(kv_lines())
        r, s = post("/v1/chat/completions", {"model": gen_model, "messages": history, "max_tokens": max_tokens,
                                             "temperature": 0, "stream": False})
        lines = kv_lines()
        return r, s, (parse_kv(lines[n]) if len(lines) > n else {})

    ra, a_s, kv_a = chat(64)
    history.append({"role": "assistant", "content": ra["choices"][0]["message"]["content"]})
    history.append({"role": "user", "content": "Et pourquoi réutiliser le préfixe accélère-t-il la réponse suivante ?"})
    # 3. chat, warm prefix, 64 tokens
    rb, b_s, kv_b = chat(64)
    history.append({"role": "assistant", "content": rb["choices"][0]["message"]["content"]})
    history.append({"role": "user", "content": "Développe en un long paragraphe, avec un exemple concret."})
    # 4. chat, warm prefix, 256 tokens (throughput)
    rc, c_s, kv_c = chat(256)

    def run(resp, wall, kv):
        u = resp.get("usage", {}); engine_ms = kv.get("total_ms")
        return {"wall_seconds": wall, "engine_ms": engine_ms,
                "queue_and_transport_ms": round(wall * 1000 - engine_ms, 1) if engine_ms is not None else None,
                "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                "cached_prefix_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "kv_reused": kv.get("reused"), "kv_gen": kv.get("gen"),
                "finish_reason": resp["choices"][0].get("finish_reason"),
                "tokens_per_engine_second": round(u.get("completion_tokens", 0) / (engine_ms / 1000), 1) if engine_ms else None}

    log_f.flush(); logtxt = open(log_path, encoding="utf-8", errors="replace").read()
    offl = re.search(r"offloaded (\d+)/(\d+) layers to GPU", logtxt)
    loaded = re.findall(r"\[CPU\] loaded: (\S+) kv=(\S+) ctx=(\d+) threads=(\d+)", logtxt)

    def pval(k):
        m = re.search(rf"^{k}:\s*(\S+)", ptxt, re.M); return m.group(1) if m else None

    # machine profile: registry and Win32 API, no external tools
    def reg(path, name):
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as k: return winreg.QueryValueEx(k, name)[0]
        except Exception: return None

    class MEM(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEM(); m.dwLength = ctypes.sizeof(MEM); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    cpu = (reg(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString") or platform.processor()).strip()
    sysmodel = " ".join(filter(None, [reg(r"HARDWARE\DESCRIPTION\System\BIOS", "SystemManufacturer"),
                                      reg(r"HARDWARE\DESCRIPTION\System\BIOS", "SystemProductName")]))
    build_info = os.path.join(a.bundle, "build-info.txt")
    toolchain = open(build_info, encoding="utf-8").read().strip() if os.path.isfile(build_info) else None

    receipt = {
      "schema": "eie.windows-receipt/v1",
      "date": time.strftime("%Y-%m-%d"),
      "provenance": "maintainer-side local execution of the distributed bundle; not independent replication",
      "server_source": a.rev,
      "runtime_base": "2168b0cd8b87c75c29a1e6588692ebbb805b9bd2",
      "runtime_patch": "patches/ews-runtime-2168b0.patch",
      "profile": {
        "arch": arch,
        "windows": f"{platform.win32_ver()[0]} {platform.win32_ver()[1]} ({platform.win32_edition()})",
        "model": sysmodel, "cpu": cpu, "logical_processors": os.cpu_count(),
        "memory_gb": round(m.ullTotalPhys / 1e9, 1),
        "toolchain": toolchain,
        "preset": a.preset, "n_ctx": pval("n_ctx"), "type_k": pval("type_k"), "type_v": pval("type_v"),
        "flash_attn": pval("flash_attn"),
        "gpu_layers_offloaded": f"{offl.group(1)}/{offl.group(2)}" if offl else "0 (CPU)",
        "effective_kv_and_ctx_per_model": [{"model": mm, "kv": kv, "ctx": int(cx), "threads": int(th)} for mm, kv, cx, th in loaded],
      },
      "sha256": {"eie-server.exe": sha(exe), **{os.path.basename(p): sha(p) for p in ggufs}},
      "load_seconds_until_all_models_healthy": load_s,
      "health": health,
      "embeddings": {"model": emb_model, "dimensions": len(vec), "l2_norm": round(norm, 4), "wall_seconds": emb_s},
      "chat": {"model": gen_model,
               "cold_prefix_64": run(ra, a_s, kv_a),
               "warm_prefix_64": run(rb, b_s, kv_b),
               "warm_prefix_256": run(rc, c_s, kv_c)},
      "limits": ["one machine, one session, warm page cache for the model files",
                 "engine_ms is the server-side request time from the [KV] log line; queue_and_transport_ms is wall minus engine",
                 "tokens_per_engine_second includes prefill of the new turn; warm_prefix_256 approximates decode rate",
                 "no comparison with llama.cpp or Ollama on the same machine", "no cold-cache protocol, no dispersion"]
    }
    json.dump(receipt, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(receipt["chat"], indent=1))
    print("receipt:", out)
    print("engine log:", log_path)
finally:
    proc.kill(); proc.wait(); log_f.close()
