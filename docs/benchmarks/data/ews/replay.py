#!/usr/bin/env python3
"""EWS P0.3 — rejeu de traces de routage MoE, conforme à KILL-THRESHOLD.md v2.

Métrique primaire: octets froids par token (C_P) pour chaque politique, par couche,
segment decode. Politiques: C0, static-hotset, freq-online, LRU, LRU-K2,
LRU-K2+prefetch bigramme (total + stall), Bélády.
Diagnostic: U_temporal. Verdict: S1 (hard floor 2 tok/s @4,5 Go/s sur Bélády),
S2 (capture < 0,5), GO-simplifié / GO-dynamique. Règle prédicteur sur C_stall.

Usage: python replay.py trace1.tsv [trace2.tsv ...] [--out rapport.md]
       [--verdict-on w1,w2,...]   # workloads comptant pour le verdict (défaut: tous sauf ceux marqués 'diag')
"""
import sys
import math
import json
from collections import defaultdict, deque

BW_GBS = 4.5                 # NVMe Gen4 effectif retenu par le seuil
TOKS_FLOOR = 2.0             # plancher produit §9 (Mixtral > 2 tok/s)
F_VERDICT = 0.25
BUDGET_FRACTIONS = [0.125, 0.25, 0.375, 0.5, 0.65]

# slab (Mo/expert/couche) par modèle — mesurés sur les GGUF locaux
SLAB_BY_MODEL = [
    ("mixtral", 114.4),        # gate 33,0 + up 33,0 + down 48,2 (Q4_K_M)
    ("gemma-4-26b", 3.345),    # slab MOYEN mesuré sur les 30 couches du GGUF réel
    ("gemma-4-26B", 3.345),
    ("qwen3-30b-a3b", 2.857),  # slab MOYEN mesuré sur le GGUF réel (P0.4-ENV.md)
]


def parse_trace(path):
    meta = {}
    per_layer = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                if "=" in line:
                    k, v = line[1:].strip().split("=", 1)
                    meta[k.strip()] = v.strip()
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            per_layer[int(parts[1])].append(tuple(int(x) for x in parts[2:]))
    return meta, per_layer


def slab_mb_for(meta):
    m = meta.get("model", "").lower()
    for key, v in SLAB_BY_MODEL:
        if key.lower() in m:
            return v
    raise SystemExit(f"slab inconnu pour modele: {meta.get('model')} — ajouter a SLAB_BY_MODEL")


def decode_segment(per_layer, n_predict):
    return {l: recs[-n_predict:] for l, recs in per_layer.items()}


def entropy(counts):
    tot = sum(counts.values())
    if tot == 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counts.values() if c > 0)


def freq_stats(seg):
    per_layer_freq, agg = {}, defaultdict(int)
    for l, recs in seg.items():
        f = defaultdict(int)
        for r in recs:
            for e in r:
                f[e] += 1
                agg[e] += 1
        per_layer_freq[l] = dict(f)
    return per_layer_freq, dict(agg)


def reuse_distances(seg):
    dists = []
    for l, recs in seg.items():
        last = {}
        for t, r in enumerate(recs):
            for e in set(r):
                if e in last:
                    dists.append(t - last[e])
                last[e] = t
    return dists


# ---- simulateurs. Tous par couche; retournent (misses_ou_octets_en_acces, n_accès) ----

def sim_lru(seg, slots, k=1):
    """LRU / LRU-K(2). Retourne (hits, tot, misses)."""
    hits = tot = 0
    for l, recs in seg.items():
        cache = {}
        for t, r in enumerate(recs):
            for e in r:
                tot += 1
                if e in cache:
                    hits += 1
                else:
                    if len(cache) >= slots:
                        victim = min(cache, key=lambda x: cache[x][0] if len(cache[x]) >= k else -math.inf)
                        del cache[victim]
                    cache[e] = deque(maxlen=k)
                cache[e].append(t)
    return hits, tot


def sim_belady(seg, slots):
    hits = tot = 0
    for l, recs in seg.items():
        nxt = defaultdict(deque)
        for t, r in enumerate(recs):
            for e in set(r):
                nxt[e].append(t)
        cache = set()
        for t, r in enumerate(recs):
            for e in set(r):
                if nxt[e] and nxt[e][0] == t:
                    nxt[e].popleft()
            for e in r:
                tot += 1
                if e in cache:
                    hits += 1
                else:
                    if len(cache) >= slots:
                        victim = max(cache, key=lambda x: nxt[x][0] if nxt[x] else math.inf)
                        cache.discard(victim)
                    cache.add(e)
    return hits, tot


def sim_static(seg, slots):
    """Hotset statique: top-freq sur 1re moitié, résident, évalué sur 2de moitié.
    Retourne (hits, tot) sur la 2de moitié + coût de préchargement amorti ignoré."""
    hits = tot = 0
    for l, recs in seg.items():
        half = len(recs) // 2
        f = defaultdict(int)
        for r in recs[:half]:
            for e in r:
                f[e] += 1
        resident = set(sorted(f, key=f.get, reverse=True)[:slots])
        for r in recs[half:]:
            for e in r:
                tot += 1
                if e in resident:
                    hits += 1
    return hits, tot


def sim_hybrid(seg, slots, pin_frac=0.5):
    """Politique P1 réellement prévue: hotset ÉPINGLÉ (top-freq de la 1re moitié,
    n_pin = max(1, round(slots*pin_frac))) + LRU simple sur les slots restants.
    Évaluée sur la 2de moitié (comme static). Retourne (hits, tot)."""
    hits = tot = 0
    for l, recs in seg.items():
        half = len(recs) // 2
        f = defaultdict(int)
        for r in recs[:half]:
            for e in r:
                f[e] += 1
        n_pin = max(1, round(slots * pin_frac))
        pinned = set(sorted(f, key=f.get, reverse=True)[:n_pin])
        n_dyn = max(0, slots - len(pinned))
        cache = {}
        for t, r in enumerate(recs[half:]):
            for e in r:
                tot += 1
                if e in pinned:
                    hits += 1
                    continue
                if e in cache:
                    hits += 1
                elif n_dyn > 0:
                    if len(cache) >= n_dyn:
                        victim = min(cache, key=lambda x: cache[x])
                        del cache[victim]
                    cache[e] = t
                if e in cache:
                    cache[e] = t
    return hits, tot


def sim_online_freq(seg, slots, slab_mb):
    """Résidence = top-freq cumulée, recalculée après chaque token.
    Octets facturés: miss à la demande (lecture transitoire) + entrées dans le
    set résident (churn). Retourne (octets_totaux_mb, n_tokens)."""
    bytes_mb = 0.0
    n_tok = min(len(r) for r in seg.values())
    for l, recs in seg.items():
        counts = defaultdict(int)
        resident = set()
        for t, r in enumerate(recs):
            for e in r:
                if e not in resident:
                    bytes_mb += slab_mb
                counts[e] += 1
            new_res = set(sorted(counts, key=counts.get, reverse=True)[:slots])
            entered = new_res - resident
            # un expert qui vient d'être lu à la demande et entre au même token
            # n'est pas refacturé
            entered -= set(r)
            bytes_mb += slab_mb * len(entered)
            resident = new_res
    return bytes_mb, n_tok


def sim_pred(seg, slots, k, budget, slab_mb):
    """LRU-K + prefetch bigramme L->L+1 online. Facture tout fetch (demande ou
    prefetch). Retourne (total_mb, stall_mb, n_tokens, prec, rec)."""
    layers = sorted(seg.keys())
    n_tok = min(len(seg[l]) for l in layers)
    caches = {l: {} for l in layers}
    big = defaultdict(lambda: defaultdict(int))
    freq = defaultdict(lambda: defaultdict(int))
    total_mb = stall_mb = 0.0
    p_hits = p_tot = r_tot = 0

    def insert(l, e, t):
        c = caches[l]
        if e in c:
            c[e].append(t)
            return
        if len(c) >= slots:
            victim = min(c, key=lambda x: c[x][0] if len(c[x]) >= k else -math.inf)
            del c[victim]
        c[e] = deque([t], maxlen=k)

    for t in range(n_tok):
        for li, L in enumerate(layers):
            # accès à la demande
            for e in seg[L][t]:
                if e in caches[L]:
                    caches[L][e].append(t)
                else:
                    total_mb += slab_mb
                    stall_mb += slab_mb
                    insert(L, e, t)
            # prefetch pour L+1 (même token), bigramme appris online
            if li + 1 < len(layers):
                L1 = layers[li + 1]
                scores = defaultdict(int)
                for e in seg[L][t]:
                    for e1, n in big[(L, e)].items():
                        scores[e1] += n
                pred = sorted(scores, key=scores.get, reverse=True)[:budget]
                if len(pred) < budget:
                    for e1 in sorted(freq[L1], key=freq[L1].get, reverse=True):
                        if e1 not in pred:
                            pred.append(e1)
                        if len(pred) == budget:
                            break
                actual = set(seg[L1][t])
                p_hits += len(set(pred) & actual)
                p_tot += max(len(pred), 1) if pred else 1
                r_tot += len(actual)
                for e1 in pred:
                    if e1 not in caches[L1]:
                        total_mb += slab_mb
                        insert(L1, e1, t)   # arrivera avant l'accès -> pas de stall
                # mise à jour bigramme APRES prédiction
                for e in seg[L][t]:
                    for e1 in seg[L1][t]:
                        big[(L, e)][e1] += 1
                for e1 in seg[L1][t]:
                    freq[L1][e1] += 1
    prec = p_hits / p_tot if p_tot else 0.0
    rec = p_hits / r_tot if r_tot else 0.0
    return total_mb, stall_mb, n_tok, prec, rec


def analyze(path):
    meta, per_layer = parse_trace(path)
    n_predict = int(meta.get("n_predict", 0))
    wl = meta.get("workload", path)
    n_expert = int(meta.get("n_expert", 8))
    top_k = int(meta.get("top_k", 2))
    slab_mb = slab_mb_for(meta)

    seg = decode_segment(per_layer, n_predict)
    n_layers = len(seg)
    n_tok = min(len(r) for r in seg.values())
    C0 = n_layers * top_k * slab_mb / 1024.0   # Go/token

    out = {"workload": wl, "n_layers": n_layers, "n_decode_tokens": n_tok,
           "n_expert": n_expert, "top_k": top_k, "slab_mb": slab_mb,
           "C0_gb_per_tok": round(C0, 3), "budgets": {}}

    plf, agg = freq_stats(seg)
    e_max = math.log2(n_expert)
    out["entropy_agg"] = round(entropy(defaultdict(int, agg)), 3)
    out["entropy_max"] = round(e_max, 3)
    # entropie NORMALISÉE (0..1): pré-filtre d'éligibilité EWS (design v0.4).
    # ~1.0 = routage uniforme (Mixtral-like), nettement <1 = hotset exploitable.
    out["entropy_norm_agg"] = round(entropy(defaultdict(int, agg)) / e_max, 4)
    out["entropy_per_layer"] = {l: round(entropy(defaultdict(int, f)), 3) for l, f in sorted(plf.items())}
    out["entropy_norm_per_layer"] = {l: round(entropy(defaultdict(int, f)) / e_max, 4) for l, f in sorted(plf.items())}

    rd = sorted(reuse_distances(seg))
    if rd:
        out["reuse_dist"] = {"p50": rd[len(rd) // 2], "p90": rd[int(len(rd) * 0.9)],
                             "mean": round(sum(rd) / len(rd), 2)}

    for F in BUDGET_FRACTIONS:
        slots = max(1, round(F * n_expert))
        acc_per_tok = n_layers * top_k

        bh, bt = sim_belady(seg, slots)
        lh, lt = sim_lru(seg, slots, 1)
        kh, kt = sim_lru(seg, slots, 2)
        sh, st = sim_static(seg, slots)
        yh, yt = sim_hybrid(seg, slots)
        H_bel, H_lru, H_lruk, H_st = bh / bt, lh / lt, kh / kt, sh / st
        H_hyb = yh / yt

        def C_of(H):   # Go/token depuis un hit rate
            return C0 * (1 - H)

        on_mb, on_tok = sim_online_freq(seg, slots, slab_mb)
        C_online = on_mb / on_tok / 1024.0

        pt_mb, ps_mb, pn, prec, rec = sim_pred(seg, slots, 2, top_k, slab_mb)
        C_pred_total = pt_mb / pn / 1024.0
        C_pred_stall = ps_mb / pn / 1024.0

        C_bel, C_lru, C_lruk, C_stat = C_of(H_bel), C_of(H_lru), C_of(H_lruk), C_of(H_st)
        C_hyb = C_of(H_hyb)
        U_temporal = (H_bel - H_st) / (1 - H_st) if H_st < 1 else 0.0
        # capture: OCTETS TOTAUX UNIQUEMENT (audit GPT v3: jamais C_stall ici —
        # un prefetch réussi réduit les stalls sans réduire les octets lus,
        # mélanger les deux flatterait artificiellement le prédicteur)
        C_v1_total = min(C_lru, C_lruk, C_pred_total, C_hyb)
        denom = C_stat - C_bel
        capture = 1.0 if denom < 0.02 * C0 else (C_stat - C_v1_total) / denom

        # concentration normalisée du routage (GPT, DESCRIPTIF uniquement, pas
        # un critère GO/KILL): part du hit statique au-delà du trivial.
        F_eff = slots / n_expert
        concentration = (H_st - F_eff) / (1 - F_eff) if F_eff < 1 else 0.0

        out["budgets"][F] = {
            "slots_per_layer": slots,
            "concentration": round(concentration, 4),
            "H": {"belady": round(H_bel, 4), "static": round(H_st, 4), "lru": round(H_lru, 4),
                  "lruk2": round(H_lruk, 4), "hybrid": round(H_hyb, 4)},
            "C_gb": {"C0": round(C0, 3), "belady": round(C_bel, 3), "static": round(C_stat, 3),
                     "online": round(C_online, 3), "lru": round(C_lru, 3), "lruk2": round(C_lruk, 3),
                     "hybrid": round(C_hyb, 3), "v1_total": round(C_v1_total, 3),
                     "pred_total": round(C_pred_total, 3), "pred_stall": round(C_pred_stall, 3)},
            "R": {p: round(1 - v / C0, 3) for p, v in
                  [("belady", C_bel), ("static", C_stat), ("lruk2", C_lruk),
                   ("hybrid", C_hyb), ("pred_stall", C_pred_stall)]},
            "toks_belady": round(BW_GBS / C_bel, 2) if C_bel > 0 else float("inf"),
            "toks_lruk2": round(BW_GBS / C_lruk, 2) if C_lruk > 0 else float("inf"),
            "toks_v1_total": round(BW_GBS / C_v1_total, 2) if C_v1_total > 0 else float("inf"),
            "toks_pred_stall": round(BW_GBS / C_pred_stall, 2) if C_pred_stall > 0 else float("inf"),
            "U_temporal": round(U_temporal, 4),
            "capture": round(capture, 4),
            "bigram": {"precision": round(prec, 4), "recall": round(rec, 4)},
            "pred_rule": bool(C_pred_stall <= 0.95 * C_lruk and C_pred_total <= 1.2 * C_lruk),
        }
    return out


def main():
    args = sys.argv[1:]
    out_md = None
    if "--out" in args:
        i = args.index("--out")
        out_md = args[i + 1]
        del args[i:i + 2]
    diag_marks = []
    if "--diag" in args:   # workloads hors verdict, ex: --diag gemma
        i = args.index("--diag")
        diag_marks = args[i + 1].split(",")
        del args[i:i + 2]
    paths = args

    results = [analyze(p) for p in paths]
    for r in results:
        r["diagnostic_only"] = any(m.lower() in r["workload"].lower() for m in diag_marks)

    lines = ["# P0.3 — Résultats du rejeu (protocole KILL-THRESHOLD, version hashée du manifest en vigueur)\n"]
    for r in results:
        tag = " *(diagnostic, hors verdict)*" if r["diagnostic_only"] else ""
        lines.append(f"## {r['workload']}{tag} — {r['n_decode_tokens']} tok decode, "
                     f"{r['n_layers']} couches, {r['n_expert']} experts top-{r['top_k']}, "
                     f"slab {r['slab_mb']} Mo, C0 = {r['C0_gb_per_tok']} Go/tok\n")
        lines.append(f"- Entropie agrégée: {r['entropy_agg']} bits (max {r['entropy_max']}) — **normalisée: {r['entropy_norm_agg']}**")
        enl = list(r["entropy_norm_per_layer"].values())
        lines.append(f"- Entropie normalisée par couche: min {min(enl):.3f}, médiane {sorted(enl)[len(enl)//2]:.3f}, max {max(enl):.3f}")
        if "reuse_dist" in r:
            d = r["reuse_dist"]
            lines.append(f"- Distance de réutilisation: p50={d['p50']}, p90={d['p90']}, moy={d['mean']}")
        lines.append("")
        lines.append("| F | slots | conc. | H_stat | H_hyb | H_LRUK | H_Bél | C_stat | C_hyb | C_LRUK | C_v1 | C_stall | C_Bél | tok/s Bél | tok/s v1 | U_temp | capture | bigr P/R | règle préd |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for F, b in r["budgets"].items():
            bg = b["bigram"]
            lines.append(
                f"| {F:.3f} | {b['slots_per_layer']} | {b['concentration']:.3f} | {b['H']['static']:.3f} | {b['H']['hybrid']:.3f} | {b['H']['lruk2']:.3f} | "
                f"{b['H']['belady']:.3f} | {b['C_gb']['static']:.2f} | {b['C_gb']['hybrid']:.2f} | {b['C_gb']['lruk2']:.2f} | "
                f"{b['C_gb']['v1_total']:.2f} | {b['C_gb']['pred_stall']:.2f} | {b['C_gb']['belady']:.2f} | {b['toks_belady']} | "
                f"{b['toks_v1_total']} | {b['U_temporal']:.3f} | {b['capture']:.3f} | "
                f"{bg['precision']:.2f}/{bg['recall']:.2f} | {'OUI' if b['pred_rule'] else 'non'} |")
        lines.append("")

    verdict_rs = [r for r in results if not r["diagnostic_only"]]
    if not verdict_rs:
        report = "\n".join(lines + ["*(run 100 % diagnostique : aucun verdict calculé)*"])
        print(report)
        if out_md:
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(report)
            with open(out_md.replace(".md", ".json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=1, ensure_ascii=False)
        return
    # S1 tri-état (audit GPT v3): partition complète des résultats possibles
    bel_toks = [r["budgets"][F_VERDICT]["toks_belady"] for r in verdict_rs]
    v1_toks = [r["budgets"][F_VERDICT]["toks_v1_total"] for r in verdict_rs]
    n_under = sum(1 for t in bel_toks if t < TOKS_FLOOR)
    if n_under == len(verdict_rs):
        s1_state = "S1-KILL"
    elif n_under > 0:
        s1_state = "S1-MIXED"
    else:
        s1_state = "S1-PASS"
    s2 = (s1_state == "S1-PASS") and all(r["budgets"][F_VERDICT]["capture"] < 0.5 for r in verdict_rs)
    # condition absolue de GO: la politique PRATIQUE (octets totaux) doit aussi
    # tenir le plancher sur chaque workload de verdict
    v1_floor_ok = all(t >= TOKS_FLOOR for t in v1_toks)
    u_temps = [r["budgets"][F_VERDICT]["U_temporal"] for r in verdict_rs]
    go = (s1_state == "S1-PASS") and (not s2) and v1_floor_ok
    go_simple = go and all(u < 0.10 for u in u_temps)
    pred_in = sum(1 for r in verdict_rs if r["budgets"][F_VERDICT]["pred_rule"])

    lines.append(f"## Verdict (F = {F_VERDICT:.0%}, workloads de verdict)\n")
    lines.append(f"- tok/s Bélády @{BW_GBS} Go/s: {bel_toks} (plancher de viabilité I/O: {TOKS_FLOOR})")
    lines.append(f"- tok/s politique v1 (C_v1_total): {v1_toks}")
    lines.append(f"- capture (octets totaux): {[r['budgets'][F_VERDICT]['capture'] for r in verdict_rs]}")
    lines.append(f"- U_temporal: {u_temps}")
    lines.append(f"- **État S1: {s1_state}** ({n_under}/{len(verdict_rs)} workloads sous le plancher Bélády)")
    lines.append(f"- **S2 (stop & redesign): {'DÉCLENCHÉ' if s2 else 'non déclenché'}**")
    lines.append(f"- Plancher pratique v1 tenu sur tous les workloads: {'oui' if v1_floor_ok else 'NON'}")
    if s1_state == "S1-KILL":
        lines.append("- **VERDICT: KILL — la prémisse v1 est morte sur ce véhicule.**")
    elif s1_state == "S1-MIXED":
        lines.append("- **VERDICT: MIXED — périmètre non généralisable; redesign ou restriction de workload, pas de GO général.**")
    elif s2:
        lines.append("- **VERDICT: STOP & REDESIGN — le potentiel existe, la politique v1 ne le capture pas.**")
    elif not v1_floor_ok:
        lines.append("- **VERDICT: STOP & REDESIGN — Bélády passe mais la politique pratique reste sous le plancher.**")
    else:
        lines.append(f"- **VERDICT: {'GO-simplifié (hotset/pinning porte le gain)' if go_simple else 'GO-dynamique'}**")
    lines.append(f"- Règle prédicteur (stall/total, indépendante) satisfaite sur {pred_in}/{len(verdict_rs)} workloads")

    report = "\n".join(lines)
    print(report)
    if out_md:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(report)
        with open(out_md.replace(".md", ".json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
