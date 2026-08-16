#!/usr/bin/env python3
"""Outil de verdict C+ (protocole VERDICT-CPLUS-PROTOCOL.md — à hasher avec lui).

Commandes:
  hotset  <trace_CAL.tsv> <out.json>
      Artefact hotset par couche, convention EXACTE du moteur gelé:
      segment = derniers n_predict records/couche, pin = top-24 par fréquence
      de la PREMIÈRE MOITIÉ du segment (CAL[0:half]).

  cstatic <trace_FULL.tsv> <hotset.json> <gguf> <n_eval>
      C_static sur EVAL (= n_eval derniers tokens/couche): résidents = hotset,
      miss = accès hors hotset; octets PHYSIQUES par miss = somme des tailles
      alignées des chunks du layout gelé (grille 512 KiB / 4 KiB sur les
      offsets réels du GGUF, padding et dernier chunk partiel inclus).

  physical <gguf>
      Table des octets physiques par (couche, expert) — contrôle.
"""
import sys
import json
import math
from collections import defaultdict

CHUNK = 512 * 1024
ALIGN = 4096
N_PIN = 24


def parse_trace(path):
    meta, per_layer = {}, defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                if "=" in line:
                    k, v = line[1:].strip().split("=", 1)
                    meta[k.strip()] = v.strip()
                continue
            p = line.split()
            if len(p) >= 3:
                per_layer[int(p[1])].append(tuple(int(x) for x in p[2:]))
    return meta, per_layer


def hotset_cmd(trace_path, out_path):
    meta, per_layer = parse_trace(trace_path)
    n_predict = int(meta["n_predict"])
    hs = {}
    for l, recs in sorted(per_layer.items()):
        seg = recs[-n_predict:]
        half = len(seg) // 2
        freq = defaultdict(int)
        for r in seg[:half]:
            for e in r:
                freq[e] += 1
        pinned = sorted(sorted(freq), key=lambda e: -freq[e])[:N_PIN]
        hs[str(l)] = sorted(pinned)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n_pin": N_PIN, "convention": "top-freq CAL[0:half], moteur gelé",
                   "source_trace_n_predict": n_predict, "hotset": hs}, f, indent=1)
    print(f"hotset écrit: {out_path} ({len(hs)} couches, pin {N_PIN})")


def physical_table(gguf_path):
    sys.path.insert(0, r"C:\Users\User\projects\llama.cpp\gguf-py")
    from gguf import GGUFReader
    r = GGUFReader(gguf_path)
    # offset absolu des données: data_offset du reader
    data_off = r.data_offset if hasattr(r, "data_offset") else None
    tensors = {t.name: t for t in r.tensors}
    table = {}
    l = 0
    while f"blk.{l}.ffn_gate_up_exps.weight" in tensors:
        gu = tensors[f"blk.{l}.ffn_gate_up_exps.weight"]
        dn = tensors[f"blk.{l}.ffn_down_exps.weight"]
        n_expert = int(gu.shape[-1]) if int(gu.shape[-1]) > 8 else int(gu.shape[0])
        gu_slab = gu.n_bytes // 128
        dn_slab = dn.n_bytes // 128
        gu_off = int(gu.data_offset)
        dn_off = int(dn.data_offset)
        per_e = []
        for e in range(128):
            phys = 0
            for off, slab in ((gu_off + e * gu_slab, gu_slab), (dn_off + e * dn_slab, dn_slab)):
                a0 = off & ~(ALIGN - 1)
                a1 = (off + slab + ALIGN - 1) & ~(ALIGN - 1)
                # grille de chunks: sum des tailles alignées = a1-a0 exactement
                phys += a1 - a0
            per_e.append(phys)
        table[l] = per_e
        l += 1
    return table


def cstatic_cmd(trace_path, hotset_path, gguf_path, n_eval):
    meta, per_layer = parse_trace(trace_path)
    n_predict = int(meta["n_predict"])
    with open(hotset_path, encoding="utf-8") as f:
        hs = json.load(f)["hotset"]
    phys = physical_table(gguf_path)
    total_miss = 0
    total_bytes = 0
    n_acc = 0
    for l, recs in sorted(per_layer.items()):
        seg = recs[-n_predict:]
        ev = seg[-n_eval:]
        resident = set(hs[str(l)])
        for r in ev:
            for e in r:
                n_acc += 1
                if e not in resident:
                    total_miss += 1
                    total_bytes += phys[l][e]
    print(f"C_static EVAL: {total_miss} miss / {n_acc} accès (hit {1 - total_miss / n_acc:.4f})")
    print(f"octets physiques: {total_bytes / 1e6:.1f} Mo total, {total_bytes / 1e6 / n_eval:.2f} Mo/token")
    print(f"(engine: C_engine = miss_EVAL × mêmes octets_physiques par (l,e) — voir protocole)")


class SlruPy:
    """Réplique EXACTE de la Slru C++ gelée (p1a2 main.cpp): free-list d'abord,
    éviction = premier non-in_use de probation puis protected, débordement de
    probation toléré tant que des slots existent, démotion avec éviction qui
    rend le slot à la free-list."""
    def __init__(self, pinned, n_slots=32, n_prob=4, n_prot=4):
        self.pinned = set(pinned)
        self.probation, self.protected = [], []      # index 0 = LRU
        self.slot_of = {}
        self.free = list(range(n_slots - 1, -1, -1))
        for e in sorted(self.pinned):                # consomme des slots comme le C++
            self.slot_of[e] = self.free.pop()
        self.n_prob, self.n_prot = n_prob, n_prot
        self.in_use = set()

    def lookup(self, e):
        return self.slot_of.get(e, -1)

    def touch(self, e):
        if e in self.pinned:
            return
        if e in self.protected:
            self.protected.remove(e)
            self.protected.append(e)
            return
        if e in self.probation:
            self.probation.remove(e)
            if self.n_prot > 0:
                self.protected.append(e)
                if len(self.protected) > self.n_prot:
                    dem = self.protected.pop(0)
                    self.probation.append(dem)
                    if len(self.probation) > self.n_prob:
                        v = self.probation.pop(0)
                        self.free.append(self.slot_of.pop(v))
            else:
                self.probation.append(e)

    def try_admit(self, e):
        if self.free:
            slot = self.free.pop()
        else:
            slot = self._evict()
            if slot < 0:
                return -1
        self.probation.append(e)
        self.slot_of[e] = slot
        return slot

    def _evict(self):
        for q in (self.probation, self.protected):
            for v in list(q):
                if v not in self.in_use:
                    q.remove(v)
                    return self.slot_of.pop(v)
        return -1


def cengine_cmd(trace_path, hotset_path, gguf_path, n_eval):
    """Simule la policy GELÉE sur la trace complète (SLRU chaud depuis le début
    du decode, hotset fixé) et compte les octets PHYSIQUES des miss sur EVAL.
    Le nombre de miss EVAL doit être IDENTIQUE au différentiel du moteur
    (miss(R1) − miss(R3)), sinon INVALID."""
    meta, per_layer = parse_trace(trace_path)
    n_predict = int(meta["n_predict"])
    with open(hotset_path, encoding="utf-8") as f:
        hs = json.load(f)["hotset"]
    phys = physical_table(gguf_path)
    sl = {int(l): SlruPy(v) for l, v in hs.items()}
    miss_eval = 0
    bytes_eval = 0
    acc_eval = 0
    for l, recs in sorted(per_layer.items()):
        seg = recs[-n_predict:]
        start_eval = len(seg) - n_eval
        s = sl[l]
        for t, r in enumerate(seg):
            s.in_use = set(r)
            for e in r:
                if s.lookup(e) >= 0:
                    s.touch(e)
                else:
                    if s.try_admit(e) < 0:
                        raise SystemExit("SLRU saturé (ne doit pas arriver)")
                    if t >= start_eval:
                        miss_eval += 1
                        bytes_eval += phys[l][e]
                if t >= start_eval:
                    acc_eval += 1
            s.in_use = set()
    print(f"C_engine (sim policy gelée) EVAL: {miss_eval} miss / {acc_eval} accès "
          f"(hit {1 - miss_eval / acc_eval:.4f})")
    print(f"octets physiques: {bytes_eval / 1e6:.1f} Mo total, {bytes_eval / 1e6 / n_eval:.2f} Mo/token")
    print("CONTRÔLE: miss_EVAL ci-dessus doit == miss(R1) − miss(R3) du moteur, sinon INVALID")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "hotset":
        hotset_cmd(sys.argv[2], sys.argv[3])
    elif cmd == "cstatic":
        cstatic_cmd(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif cmd == "cengine":
        cengine_cmd(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif cmd == "physical":
        t = physical_table(sys.argv[2])
        import statistics
        allv = [v for pe in t.values() for v in pe]
        print(f"{len(t)} couches × 128 experts; octets physiques/expert: "
              f"min {min(allv)}, max {max(allv)}, moyen {statistics.mean(allv):.0f}")
    else:
        print(__doc__)
