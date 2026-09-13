#!/bin/bash
# EIE — reçu de validation macOS : lance le moteur d'un bundle sur un port de
# test, mesure santé / embeddings / chat à préfixe froid puis chaud, et écrit
# un JSON au format des reçus du dépôt (docs/benchmarks/data/).
# Usage : bash receipt-macos.sh --bundle DIR --models DIR [--port 8091] [--out FILE.json] [--rev GITREV]
# Ne JAMAIS le lancer sur un moteur en service : il démarre sa propre instance.
set -e
PORT=8091; OUT=""; REV="unknown"; BUNDLE=""; MODELS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --bundle) BUNDLE="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --rev) REV="$2"; shift 2 ;;
    *) echo "argument inconnu : $1"; exit 1 ;;
  esac
done
[ -x "$BUNDLE/eie-server" ] || { echo "--bundle DIR doit contenir eie-server"; exit 1; }
[ -d "$MODELS" ] || { echo "--models DIR doit contenir les GGUF"; exit 1; }
ARCH="$(uname -m)"
case "$ARCH" in arm64) PRESET="macos-silicon.yaml" ;; *) PRESET="macos-cpu.yaml" ;; esac
[ -n "$OUT" ] || OUT="receipt-macos-$ARCH-$(date +%Y%m%d).json"
WORK="$(mktemp -d)"
cp "$BUNDLE/presets/$PRESET" "$WORK/preset.yaml"
sed -i '' "s/^port:.*/port: $PORT/" "$WORK/preset.yaml"
sed -i '' "s|^model_dir:.*|model_dir: $MODELS|" "$WORK/preset.yaml"
LOG="$WORK/engine.log"
"$BUNDLE/eie-server" --config "$WORK/preset.yaml" > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
T0=$(python3 -c 'import time;print(time.time())')
for i in $(seq 1 120); do
  curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"models":2' && break; sleep 1
done
LOAD_S=$(python3 -c "import time;print(round(time.time()-$T0,2))")
HEALTH=$(curl -s -m 5 "http://127.0.0.1:$PORT/health")
echo "moteur prêt en ${LOAD_S}s : $HEALTH"

python3 - "$PORT" "$LOG" "$OUT" "$REV" "$BUNDLE" "$MODELS" "$PRESET" "$LOAD_S" "$HEALTH" << 'PY'
import sys, json, time, re, subprocess, hashlib, urllib.request, os, glob, platform
port, log, out, rev, bundle, models, preset, load_s, health = sys.argv[1:10]
base = f"http://127.0.0.1:{port}"
def post(path, body):
    req = urllib.request.Request(base+path, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=600)); return r, round(time.time()-t0, 3)
def kv_lines():
    return [l for l in open(log, errors="replace") if l.startswith("[KV]")]
def parse_kv(line):
    return {k: int(v) for k, v in re.findall(r"(reused|gen|total_ms)=(-?\d+)", line)}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
def sysctl(k):
    try: return subprocess.check_output(["sysctl", "-n", k], text=True).strip()
    except Exception: return ""
gen_model = next((os.path.basename(p)[:-5] for p in glob.glob(models+"/*.gguf") if "bge" not in p and "embed" not in p), None)
emb_model = next((os.path.basename(p)[:-5] for p in glob.glob(models+"/*.gguf") if "bge" in p or "embed" in p), None)

# 1. embeddings
emb, emb_s = post("/v1/embeddings", {"model": emb_model, "input": "Elyne vit localement dans ce Mac."})
vec = emb["data"][0]["embedding"]; norm = sum(x*x for x in vec) ** 0.5

# 2. chat, préfixe froid (contexte fixe ~250 tokens)
system = ("Tu es un assistant de test. " * 20).strip()
history = [{"role": "system", "content": system},
           {"role": "user", "content": "Explique en trois phrases ce qu'est un cache KV dans un modèle de langage."}]
n_before = len(kv_lines())
a, a_s = post("/v1/chat/completions", {"model": gen_model, "messages": history, "max_tokens": 64, "temperature": 0, "stream": False})
kv_a = parse_kv(kv_lines()[n_before]) if len(kv_lines()) > n_before else {}
history.append({"role": "assistant", "content": a["choices"][0]["message"]["content"]})
history.append({"role": "user", "content": "Et pourquoi réutiliser le préfixe accélère-t-il la réponse suivante ?"})

# 3. chat, préfixe chaud, 64 tokens
n_before = len(kv_lines())
b, b_s = post("/v1/chat/completions", {"model": gen_model, "messages": history, "max_tokens": 64, "temperature": 0, "stream": False})
kv_b = parse_kv(kv_lines()[n_before]) if len(kv_lines()) > n_before else {}
history.append({"role": "assistant", "content": b["choices"][0]["message"]["content"]})
history.append({"role": "user", "content": "Développe en un long paragraphe, avec un exemple concret."})

# 4. chat, préfixe chaud, 256 tokens (débit)
n_before = len(kv_lines())
c, c_s = post("/v1/chat/completions", {"model": gen_model, "messages": history, "max_tokens": 256, "temperature": 0, "stream": False})
kv_c = parse_kv(kv_lines()[n_before]) if len(kv_lines()) > n_before else {}

def run(resp, wall, kv):
    u = resp.get("usage", {})
    engine_ms = kv.get("total_ms")
    return {"wall_seconds": wall, "engine_ms": engine_ms,
            "queue_and_transport_ms": round(wall*1000 - engine_ms, 1) if engine_ms is not None else None,
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "cached_prefix_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "kv_reused": kv.get("reused"), "kv_gen": kv.get("gen"),
            "finish_reason": resp["choices"][0].get("finish_reason"),
            "tokens_per_engine_second": round(u.get("completion_tokens", 0) / (engine_ms/1000), 1) if engine_ms else None}

logtxt = open(log, errors="replace").read()
metal = re.search(r"ggml_metal_init: found device: (.*)", logtxt)
offl = re.search(r"offloaded (\d+)/(\d+) layers to GPU", logtxt)
loaded = re.findall(r"\[CPU\] loaded: (\S+) kv=(\S+) ctx=(\d+) threads=(\d+)", logtxt)
ptxt = open(bundle+"/presets/"+preset).read()
def pval(k):
    m = re.search(rf"^{k}:\s*(\S+)", ptxt, re.M); return m.group(1) if m else None

receipt = {
  "schema": "eie.macos-receipt/v1",
  "date": time.strftime("%Y-%m-%d"),
  "provenance": "maintainer-side local execution of the distributed bundle; not independent replication",
  "server_source": rev,
  "runtime_base": "2168b0cd8b87c75c29a1e6588692ebbb805b9bd2",
  "runtime_patch": "patches/ews-runtime-2168b0.patch",
  "profile": {
    "arch": platform.machine(), "macos": platform.mac_ver()[0],
    "model": sysctl("hw.model"), "cpu": sysctl("machdep.cpu.brand_string"),
    "memory_gb": round(int(sysctl("hw.memsize") or 0) / 1e9, 1),
    "preset": preset, "n_ctx": pval("n_ctx"), "type_k": pval("type_k"), "type_v": pval("type_v"),
    "flash_attn": pval("flash_attn"),
    "metal_device": metal.group(1).strip() if metal else None,
    "gpu_layers_offloaded": f"{offl.group(1)}/{offl.group(2)}" if offl else "0 (CPU)",
    "effective_kv_and_ctx_per_model": [{"model": m, "kv": kv, "ctx": int(cx), "threads": int(th)} for m, kv, cx, th in loaded],
  },
  "sha256": {"eie-server": sha(bundle+"/eie-server"), **{os.path.basename(p): sha(p) for p in sorted(glob.glob(models+"/*.gguf"))}},
  "load_seconds_until_two_models_healthy": float(load_s),
  "health": json.loads(health),
  "embeddings": {"model": emb_model, "dimensions": len(vec), "l2_norm": round(norm, 4), "wall_seconds": emb_s},
  "chat": {"model": gen_model,
           "cold_prefix_64": run(a, a_s, kv_a),
           "warm_prefix_64": run(b, b_s, kv_b),
           "warm_prefix_256": run(c, c_s, kv_c)},
  "limits": ["one machine, one session, warm page cache for the model files",
             "engine_ms is the server-side request time from the [KV] log line; queue_and_transport_ms is wall minus engine",
             "tokens_per_engine_second includes prefill of the new turn; warm_prefix_256 approximates decode rate",
             "no comparison with llama.cpp or Ollama on the same machine", "no cold-cache protocol, no dispersion"]
}
json.dump(receipt, open(out, "w"), indent=2, ensure_ascii=False)
print(json.dumps(receipt["chat"], indent=1))
print("reçu :", out)
PY
