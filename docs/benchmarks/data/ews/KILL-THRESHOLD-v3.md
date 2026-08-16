# P0.3-bis — Seuil de kill v3 (re-verdict sur véhicule re-scopé, FINAL)

**Statut : v3 finale post-audit GPT (7 corrections intégrées, aucune ne rend le
verdict plus facile : holdouts, tri-état S1, plancher pratique sur chaque
workload, capture sur octets totaux). En attente du go d'Alex ; au go, ce
fichier et le manifest complet sont hashés dans HASHES.txt AVANT le premier
token de trace.** Toute modification postérieure au hash sera annotée.

## Contexte

Le verdict v2 reste acquis et intouché : S1 DÉCLENCHÉ sur Mixtral (seuil hashé
avant traces, kill honoré, cause documentée). Mixtral est contrôle négatif
publié. La trace Gemma du P0.3 et le rapport Qwen P0.4 sont des diagnostics :
**P0.4 fait l'objet d'un rapport indépendant et n'intervient ni dans les seuils
ni dans le calcul du verdict Gemma v3.**

## Véhicule et périmètre

- **Véhicule de verdict : Gemma 4 26B-A4B Instruct Q4_0** —
  `google_gemma-4-26B-A4B-it-Q4_0.gguf`, 14 439 363 584 octets,
  SHA-256 `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`.
  Métadonnées lues du fichier : **30 blocs, tous MoE**, 128 experts/couche,
  top-8, expert_ff 704, gate_up fusionné (Q4_0) + down (Q4_0), total experts
  12,846 Go, **slab moyen mesuré 3,345 Mo** ⇒ **C0 = 30 × 8 × 3,345 Mo =
  0,8029 Go/token**.
- Périmètre v1 : *EWS v1 cible les MoE à granularité fine présentant une
  concentration de routage empiriquement exploitable. L'éligibilité d'un modèle
  est déterminée par profilage, non par famille ou nombre d'experts.*
- K3-class : cible crédible à profiler, PAS validée par ce document.

## Entrées : holdouts stricts

**Les entrées G1–G4 sont distinctes des entrées utilisées lors de la trace
diagnostique P0.3 et de toute trace antérieure (Mixtral W1–W4, Qwen Q1–Q2).
Aucun prompt, document source ou séquence n'est réutilisé. Les quatre fichiers
sont figés et hashés avant la première trace.**

- G1 `g1_code_holdout.txt` : analyse de deux programmes C++ jamais tracés
  (traceur P0.3 + bench P0.1) — n_predict 512, n_ctx 8192
- G2 `g2_prose_holdout.txt` : récit français inédit (l'horlogère) — 512, 4096
- G3 `g3_nmm_holdout.txt` : 25 questions d'évaluation inédites (proxy NMM) — 512, 4096
- G4 `g4_longctx_holdout.txt` : extrait `llama-model.cpp` (~8k tokens, source
  jamais utilisée ; W4 utilisait `llama-graph.cpp`) — 256, 10240

Decode greedy argmax (déterministe, sans seed), n_gpu_layers=24, params loggés
dans l'en-tête de chaque trace.

## Protocole de mesure

- Verdict décode-only ; budget PAR COUCHE (slots = F × 128) ; courbe complète
  F ∈ {12,5 ; 25 ; 37,5 ; 50 ; 65 %} ; **verdict lu à F = 25 %** (32 slots/couche).
- Politiques simulées (octets froids par token, prefetch facturé au fetch) :
  - `C0` : sans cache = 0,8029 Go/token
  - `C_static` : hotset top-fréquence appris sur la 1re moitié du segment
    decode, résident, évalué sur la 2de moitié
  - `C_hybrid` : **politique P1 réellement prévue** — hotset épinglé
    (n_pin = slots/2, appris 1re moitié) + LRU simple sur les slots restants,
    évaluée sur la 2de moitié
  - `C_online`, `C_LRU`, `C_LRUK` (K=2) : caches dynamiques purs, segment entier
  - `C_pred_total` / `C_pred_stall` : LRU-K + prefetch bigramme L→L+1 online,
    budget top_k ; total = tout octet lu, stall = miss non couverts (bloquants)
  - `C_Bélády` : oracle clairvoyant, borne physique
- `C_v1_total = min(C_LRU, C_LRUK, C_pred_total, C_hybrid)` — **octets totaux
  uniquement, jamais C_stall** (un prefetch réussi réduit les stalls sans
  réduire les octets lus ; mélanger flatterait le prédicteur).
- BW de référence : **4,5 Go/s** (conservatrice ; P0.1 mesure 6,6 Go/s sur
  slabs 3,35 Mo — l'écart est une marge, pas un paramètre ajustable).
- Le « 2,0 tok/s » ci-dessous est un **plancher de viabilité I/O : nécessaire,
  mais non suffisant pour la cible produit de 2 tok/s end-to-end**, qui sera
  validée en P1 avec calcul FFN, synchro routeur, H→D, SHA et bookkeeping.

## Verdict (partition complète, pré-enregistrée)

**État S1** — sur `tok/s_Bélády = 4,5 ÷ C_Bélády(F=25 %)` par workload :
- **S1-KILL** : < 2,0 sur les 4 workloads ⇒ prémisse EWS-v1 morte sur ce
  véhicule ; les deux kills (Mixtral, Gemma) sont publiés.
- **S1-MIXED** : < 2,0 sur ≥ 1 workload mais pas les 4 ⇒ périmètre non
  généralisable ; redesign ou restriction de workload documentée, pas de GO général.
- **S1-PASS** : ≥ 2,0 sur les 4 ⇒ passage à S2.

**S2 (stop & redesign)** — si S1-PASS :
`capture = (C_static − C_v1_total) / (C_static − C_Bélády)` par workload
(garde-fou : si `C_static − C_Bélády < 2 % de C0`, capture := 1).
S2 déclenché si capture < 0,5 sur les 4 workloads ⇒ le potentiel existe, la
politique v1 ne le capture pas ; redesign avant P1.

**Condition absolue de GO** : `4,5 ÷ C_v1_total ≥ 2,0 tok/s` sur **chacun** des
4 workloads. Si Bélády passe mais la politique pratique reste sous le plancher
sur un workload : STOP & REDESIGN, pas GO.

**GO** — si S1-PASS, S2 non déclenché, condition absolue tenue. Avec
`U_temporal = (H_Bélády − H_static) / (1 − H_static)` à F = 25 % :
- **GO-simplifié** si U_temporal < 0,10 sur les 4 workloads ⇒ P1 sans cache
  dynamique complexe : hotset/pinning calibré + éviction simple.
- **GO-dynamique** sinon (≥ 0,10 sur au moins un workload) ⇒ P1 avec part
  dynamique (LRU simple d'abord ; LRU-K/prefetch seulement sur gain démontré).

**Règle prédicteur (indépendante du verdict)** : le bigramme entre en v1 ssi
`C_pred_stall ≤ 0,95 × C_LRUK` **et** `C_pred_total ≤ 1,20 × C_LRUK` à F = 25 %
sur une majorité de workloads. Précision/rappel publiés à titre informatif.

**S3 — publiés quel que soit le verdict** : courbes C_P(F) complètes par
workload et par politique, distributions et entropies normalisées par couche
(médiane, P75, P90, proportion de couches < 0,85), distances de réutilisation,
tok/s_max par politique, comparatif final Mixtral/Gemma/Qwen.

## Manifest à hasher AVANT le premier token (dans HASHES.txt)

1. Ce fichier (`KILL-THRESHOLD-v3.md`)
2. Les 4 entrées `g*_holdout.txt`
3. `replay.py` (version courante, avec sim_hybrid et verdict tri-état)
4. `tracer/main.cpp` (source du traceur)
5. SHA-256 du GGUF Gemma (ci-dessus)
6. Commit runtime : llama.cpp `fae3a28070fe4026f87bd6a544aba1b2d1896566`
   (build MSVC 14.44.35207 Release CUDA)
7. Paramètres de génération (section Entrées)
8. Métadonnées GGUF réellement lues (section Véhicule)
