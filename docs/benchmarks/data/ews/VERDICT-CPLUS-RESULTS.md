# VERDICT C+ END-TO-END — RÉSULTATS (2026-08-16)

Protocole : `VERDICT-CPLUS-PROTOCOL.md` (SHA 1d530b8d…) + `AMENDMENT-CPLUS-v1.1.md`
(SHA 6727ad54…, incident P1-C+-CTX-01). Moteur : candidat gelé `ENGINE-FREEZE.md`
(binaire 15f43583…). Holdouts : G5'/G6/G7/G8' hashés avant toute donnée.

## Contrôles INVALID — tous passés

- Identité R1/R2 exacte (routage, hits, misses, cold) : 4/4 holdouts ✓
- |T_other(R1) − T_other(R2)| : 0,1–0,7 ms < 3 ms : 4/4 ✓ ; T_other ≥ 0 : 4/4 ✓
- **Identité miss_EVAL : réplique SLRU vs différentiel moteur (R1−R3), à l'unité
  près : 12 191=12 191 ; 7 413=7 413 ; 15 583=15 583 ; 5 305=5 305** ✓✓

## Critères

**C1 — viabilité (pire des deux runs, EVAL) : PASS 4/4**

| | G5' code | G6 prose | G7 NMM | G8' ctx-max |
|---|---|---|---|---|
| tok/s EVAL (pire) | 7,19 | 11,10 | 6,55 | 7,60 |
| plancher | 2,0 | 2,0 | 2,0 | 2,0 |

**C2 — non-régression (octets physiques EVAL) : PASS 4/4**

| Mo/token EVAL | G5' | G6 | G7 | G8' |
|---|---|---|---|---|
| C_engine | 159,7 | 97,1 | 204,1 | 139,0 |
| C_static | 266,9 | 181,8 | 396,6 | 245,3 |

**C3 — apport dynamique strict (≥ 2 requis) : PASS 4/4** — la SLRU réduit les
octets froids de 40 à 48 % par rapport au hotset pur sur CHAQUE holdout.

## VERDICT : **PASS** (C1 ∧ C2 ∧ C3)

**Portée exacte, ni plus ni moins** : le candidat EWS gelé (policy 75/25,
F=25 %, chunks 512 KiB, SHA-256/chunk fail-closed) généralise sur quatre
holdouts end-to-end inédits, sur Gemma-4-A4B, **borné à n_ctx ≤ 4096**
(limitation documentée du candidat). Ne valide ni le contexte long, ni
K3-class, ni d'autres modèles, ni le format d'index final.

## Lectures notables (recherche, non bloquantes)

- L'apport dynamique — la question qui avait tué la policy v1 au verdict v3 —
  est tranché en conditions réelles : avec une calibration courte (128 tokens),
  le hotset statique se dégrade hors développement (hit 0,51–0,77) et la SLRU
  compense (hit 0,75–0,88). **Le cache dynamique gagne sa place précisément
  là où le statique seul s'effondre.**
- G7 (questions variées) reste le régime le plus hostile : hit 0,746,
  204 Mo/token — et 6,55 tok/s quand même (3,3× le plancher).
- Écart dev→holdouts : 12,6 → 6,6-11,1 tok/s. Le moteur ralentit sur données
  neuves (working sets plus larges) mais reste 3,3–5,6× au-dessus du plancher.
- Incident P1-C+-CTX-01 au dossier : G5/G8 originaux INVALID (n_ctx), G6/G7
  acquis au premier passage, remplacements par troncature mécanique hashée.

## Chaîne de custody complète

moteur gelé → gate C+ gelé → protocole gelé → outil gelé → classes gelées →
contenus hashés → CAL seul → hotsets hashés → runs → EVAL après coup →
contrôles INVALID → verdict mécanique. Chaque étape antérieure aux données
qu'elle gouverne.
