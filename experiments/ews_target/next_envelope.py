"""Real Next composition on a copied state, concurrently with EIE EWS.

Private output includes original conversation-derived context. Keep it under build-*/.
Only the test-owned processes and the copied state are mutated.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]

def http(url, data=None, timeout=300):
    request = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

def tree_digest(root):
    sha = hashlib.sha256()
    count = 0
    for folder in ("memory", "proofs", "user-files"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file():
                sha.update(str(path.relative_to(root)).encode())
                sha.update(hashlib.sha256(path.read_bytes()).digest())
                count += 1
    return {"files": count, "sha256": sha.hexdigest()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--next", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--explicit-tools", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.state.resolve() == (args.next / "elyne_state").resolve():
        raise ValueError("This measurement requires a copied state, never production")
    for port in (18123, 18124, 18080):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Measurement port {port} already in use")
    sys.path.insert(0, str(args.next))
    import launch_ui as launch
    from elyne.runtime.resident import ResidentConfig
    from elyne.runtime.embedding_resident import EmbeddingResidentConfig
    original = tree_digest(args.next / "elyne_state")
    report = {"scope": "real Next code and copied state; separate EIE 26B service, no automatic routing",
              "original_before": original, "copied_state_before": tree_digest(args.state),
              "next_context_configured": 16384, "ews_slots": args.slots, "turns": [], "resident_calls": [], "ews_requests": []}
    report["explicit_tools"] = args.explicit_tools
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["next_commit"] = subprocess.check_output(["git", "-C", str(args.next), "rev-parse", "HEAD"], text=True).strip()
    logdir = launch.setup_runtime_logging(str(args.out / "logs"))
    workspace = args.state / "workspace"; workspace.mkdir(exist_ok=True)
    policy = launch.build_tool_authority_policy(code_dir=args.next / "elyne", desktop_dir=Path(r"C:\Users\User\Desktop"),
        dropbox_dir=Path(r"C:\Users\User\Dropbox"), user_files_dir=args.state / "user-files", workspace_dir=workspace)
    authority = launch.ElyneAuthority(
        resident_config=ResidentConfig(binary_path=launch.LLAMA, gguf_path=launch.GEMMA, model_alias="resident",
            port=18123, context_tokens=16384, mmproj_path=launch.MMPROJ),
        embedding_config=EmbeddingResidentConfig(binary_path=launch.LLAMA, gguf_path=launch.EMBEDDER,
            model_alias="nomic-embed-text-v2-moe", port=18124, expected_dimensions=768),
        environment=launch._resident_env(), resident_startup_log_directory=logdir,
        memory_root=str(args.state / "memory"), proof_root=str(args.state / "proofs"),
        gate_policy=launch.make_initiative_gate_policy(policy_id="elyne-initiative-gate", revision=1, opt_in=True,
            cooldown_seconds=10800, quota_limit=2, quota_window_seconds=86400),
        consolidation_policy=launch.make_consolidation_policy(policy_id="elyne-consolidation", revision=2,
            opt_in=True, min_new_events=4, idle_seconds=180, min_interval_seconds=900),
        belief_reflection_policy=launch.make_belief_reflection_policy(policy_id="elyne-rest-belief-reflection", revision=1, opt_in=True),
        idle_threshold_seconds=300, elyne_rest_poll_seconds=30, tool_authority_policy=policy)
    process = None
    stop = threading.Event()
    samples = []
    def gpu_sampler():
        while not stop.is_set():
            result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True)
            samples.append({"time": time.time(), "memory_mib": result.stdout.strip()})
            stop.wait(1)
    sampler = threading.Thread(target=gpu_sampler, daemon=True)
    sampler.start()
    def mark(stage):
        print(stage, flush=True)
        report.setdefault("stages", []).append({"stage": stage, "time": time.time()})
    def save(name, value):
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    def turn(label, prompt):
        mark("Next " + label)
        start = time.monotonic()
        call_start = len(report["resident_calls"])
        response = authority.message("ews-envelope-" + uuid.uuid4().hex, prompt)
        save(label + ".private.json", response)
        report["turns"].append({"label": label, "status": response["status"], "seconds": time.monotonic() - start,
                                "tool_calls": [t for call in report["resident_calls"][call_start:] for t in call["tool_calls"]],
                                "response_sha256": hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()})
        return response
    try:
        mark("boot Next")
        authority.start()
        report["runtime_before"] = authority._session._runtime_snapshot()
        save("beliefs-before.private.json", authority.transparency("beliefs"))
        resident = authority._session._resident
        # Observe request sizes and latency without changing messages, sampling or tools.
        for method_name in ("chat", "chat_cancellable"):
            original_method = getattr(resident, method_name)
            def observer(messages, *pos, _original=original_method, _name=method_name, **kw):
                start = time.monotonic()
                response = _original(messages, *pos, **kw)
                report["resident_calls"].append({"method": _name, "messages": len(messages),
                    "message_json_chars": len(json.dumps(messages, ensure_ascii=False)), "tools": len(kw.get("tools") or []),
                    "seconds": time.monotonic() - start, "completion_tokens": response.get("completion_tokens"),
                    "tool_calls": [t.get("function", {}).get("name") for t in response.get("tool_calls") or []]})
                return response
            setattr(resident, method_name, observer)
        turn("before", "Pour ce test technique, exécute maintenant inspect_runtime et résume son résultat en deux phrases. Ne te contente pas de ta connaissance antérieure." if args.explicit_tools else
             "Nous faisons un test de charge technique. Vérifie ton runtime réel et donne-moi brièvement les capacités actuellement disponibles, sans révéler d'information personnelle.")
        config = args.out / "eie.yaml"
        config.write_text(f"host: 127.0.0.1\nport: 18080\nauto_discover: false\ntype_k: f16\ntype_v: f16\nn_ctx: 2048\nflash_attn: true\nmodels:\n  gemma-26b-ews: {args.model}\news_slots:\n  gemma-26b-ews: {args.slots}\npreload: [gemma-26b-ews]\n", encoding="utf-8")
        env = dict(os.environ)
        env["PATH"] = str(ROOT / "build-ews/bin/Release") + os.pathsep + env.get("PATH", "")
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        mark("boot EIE 26B")
        with (args.out / "eie.log").open("wb") as log:
            process = subprocess.Popen([str(ROOT / "build-ews/Release/eie-server.exe"), "--config", str(config)],
                env=env, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("EIE exited during loading; see eie.log")
                try:
                    models = http("http://127.0.0.1:18080/v1/models", timeout=3)
                    if not any(m["id"] == "gemma-26b-ews" for m in models.get("data", [])):
                        raise RuntimeError("EIE is up but the EWS model failed to load")
                    break
                except (OSError, ValueError):
                    time.sleep(1)
            else: raise TimeoutError("EIE startup")
            report["vram_both_loaded"] = http("http://127.0.0.1:18080/v1/admin/vram/status")
            workload_stop = threading.Event()
            def workload():
                for index in range(12):
                    if workload_stop.is_set(): break
                    start = time.monotonic()
                    result = http("http://127.0.0.1:18080/v1/chat/completions", {"model": "gemma-26b-ews",
                        "messages": [{"role": "user", "content": f"Écris une fonction Python de cache LRU borné, puis explique ses invariants. Variante {index}."}],
                        "temperature": 0, "max_tokens": 192, "one_shot": index % 2 == 0})
                    report["ews_requests"].append({"seconds": time.monotonic() - start, "usage": result.get("usage"),
                        "nonempty": bool(result["choices"][0]["message"]["content"]), "start": time.time() - (time.monotonic() - start),
                        "end": time.time()})
                    print(f"EIE completed request {index}", flush=True)
            mark("concurrent inference")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(workload)
                try:
                    turn("during-memory", "Exécute maintenant search_memory pour rechercher EIE et EWS. Rapporte brièvement deux faits trouvés et leurs sources. Ce test nécessite l'appel réel de l'outil, pas une réponse à partir du contexte." if args.explicit_tools else
                         "Recherche dans ta mémoire ce que nous avons déjà établi techniquement sur EIE et EWS. Distingue les résultats mesurés des objectifs. Réponds brièvement avec tes sources, sans donnée personnelle.")
                    turn("during-beliefs", "Exécute search_beliefs sur preuve et hypothèse, puis utilise inspect_belief sur un des identifiants trouvés. Rapporte son numéro de version. Ne crée et ne modifie aucune croyance." if args.explicit_tools else
                         "Consulte tes croyances existantes sur la distinction entre preuve et hypothèse, puis résume en trois phrases ce que tu retiens. Ne crée aucune nouvelle croyance pour ce test.")
                    turn("during-files", "Exécute list_files pour lister le dossier racine de code. Donne trois noms réellement retournés par cet outil. N'invente aucun fichier et ne lis pas de document personnel." if args.explicit_tools else
                         "Liste les racines de fichiers auxquelles tu as réellement accès, puis les fichiers du dossier code. Donne seulement le nombre et trois noms, sans lire de document personnel.")
                finally:
                    workload_stop.set()
                    future.result(timeout=300)
            report["ews_final"] = http("http://127.0.0.1:18080/v1/admin/ews/status")
            report["runtime_during"] = authority._session._runtime_snapshot()
            save("tools.private.json", authority.transparency("tools"))
            process.terminate(); process.wait(timeout=30); process = None
        mark("EIE unloaded")
        turn("after", "Le test est terminé. Exécute à nouveau inspect_runtime et confirme son état en deux phrases." if args.explicit_tools else
             "Le test de charge est terminé. Vérifie que ton runtime, ta mémoire et tes outils sont toujours disponibles, puis réponds en deux phrases.")
        report["runtime_after"] = authority._session._runtime_snapshot()
        save("beliefs-after.private.json", authority.transparency("beliefs"))
        report["completed"] = True
        report["tools_exercised"] = sorted({t for call in report["resident_calls"] for t in call["tool_calls"]})
        report["requested_tool_paths_observed"] = {t: t in report["tools_exercised"] for t in
            ("inspect_runtime", "search_memory", "search_beliefs", "inspect_belief", "list_files")}
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate(); process.wait(timeout=30)
        try: authority.stop()
        except Exception as exc: report["stop_error"] = repr(exc)
        stop.set(); sampler.join(timeout=10)
        report["original_after"] = tree_digest(args.next / "elyne_state")
        report["original_unchanged"] = report["original_after"] == original
        save("gpu_samples.json", samples)
        save("report.json", report)
        mark("measurement stopped; production state untouched=" + str(report["original_unchanged"]))

if __name__ == "__main__":
    main()
