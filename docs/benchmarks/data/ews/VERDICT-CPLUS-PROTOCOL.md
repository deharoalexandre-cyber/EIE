# Verdict C+ end-to-end sur G5–G8 — protocole FINAL (audit GPT intégré)

**Statut : final post-audit GPT (7 clarifications, aucune ne facilite le test).
À hasher avec `verdict_cplus.py` AVANT la création des holdouts. Séquence
signée : protocole final → hash → création G5–G8 (classes ci-dessous) → hash
holdouts → traces CAL uniquement → hash artefacts hotset → 2×runs moteur (+run
CAL différentiel) → trace/replay EVAL → application mécanique.**

## Objet et portée

Le moteur gelé (`ENGINE-FREEZE.md`, référence dev 12,6 tok/s) passe-t-il le
gate C+ sur quatre workloads jamais vus ? **Portée sécurité** : un PASS valide
le pipeline fail-closed tel que testé (SHA/chunk, READY après validation
complète) — PAS le format final anti-recomposition inter-modèles/inter-versions
(binding model_id/version : `eie-ews-pack`, hors périmètre de ce verdict).

## Classes des holdouts (figées AVANT création des contenus)

Miroir de G1–G4, format Gemma, greedy argmax, n_gpu_layers=24. Aucun matériau
réutilisé de W1–W5, G1–G4, Q1–Q2 ni des prompts de dev :

| | Classe | n_predict | n_ctx |
|---|---|---|---|
| G5 | code C++ (source jamais employée) | 512 | 8192 |
| G6 | prose française inédite | 512 | 4096 |
| G7 | batterie de questions inédite (proxy NMM) | 512 | 4096 |
| G8 | contexte long ~8k (source jamais employée) | 256 | 10240 |

## CAL / EVAL

- **CAL** = première moitié du decode (256 tok pour G5-G7, 128 pour G8) ;
  **EVAL** = seconde moitié. Le hotset est appris EXCLUSIVEMENT sur CAL
  (convention du moteur gelé : top-freq des 128 premiers tokens de CAL —
  CAL[0:half], strictement inclus dans CAL). C1/C2/C3 se calculent
  EXCLUSIVEMENT sur EVAL.
- **État du cache à la frontière (convention unique)** : hotset fixé depuis
  CAL ; le moteur repart du début (prefill + decode complet) avec ce hotset ;
  le SLRU se réchauffe librement pendant CAL et **conserve son état en entrant
  dans EVAL**. La référence C_static reproduit la même convention (résidents
  statiques = hotset CAL, évalués sur EVAL).

## Ordre d'exécution anti-fuite (par holdout)

1. **Trace CAL uniquement** : traceur gelé, n_predict = |CAL|. Personne ne voit
   le routage EVAL avant les runs moteur.
2. **Artefact hotset** : `verdict_cplus.py hotset` sur la trace CAL → JSON par
   couche ; **hashé avant tout run moteur**.
3. **Runs moteur** (binaire gelé, calibré sur la trace CAL) :
   R1 = decode |CAL|+|EVAL| ; R2 = R1 identique (contrôle de déterminisme) ;
   R3 = decode |CAL| seul (pour la mesure différentielle).
4. **Après les runs seulement** : trace complète (CAL+EVAL) → références
   simulées sur EVAL.

## Mesure différentielle EVAL (le moteur gelé n'est pas modifié)

Pour toute grandeur cumulative X (temps de decode, misses, stall, etc.) :
`X_EVAL = X(R1) − X(R3)`. Donc `tok/s_EVAL = |EVAL| / (T_decode(R1) −
T_decode(R3))`, `miss_EVAL = miss(R1) − miss(R3)`, etc.

## Comptage des octets (physiques, identiques des deux côtés)

`C_engine` et `C_static` dérivent du MÊME primitif :
`cold_bytes = Σ_miss octets_physiques(layer, expert)` où octets_physiques =
somme des tailles alignées des chunks du layout gelé (grille 512 KiB alignée
4 KiB sur les régions gate_up et down, padding et dernier chunk partiel inclus,
offsets réels du GGUF). Calculés par `verdict_cplus.py` depuis le GGUF.
`C_engine` utilise miss_EVAL du moteur ; `C_static` les miss du hotset pur
simulé sur EVAL. Le compteur logique interne du moteur est informatif seulement.

## Règles de décision des deux runs

- Grandeurs déterministes (routage, hits, misses, cold, texte généré) :
  **identité EXACTE exigée entre R1 et R2**, sinon **INVALID** (pas FAIL).
- **C1 utilise le PIRE tok/s_EVAL des deux runs** (jamais moyenne ni meilleur).
- Fermeture comptable, gravée :
  `residual = |T_total − (T_compute + T_visibility_policy + T_routing_stall +
  T_graph_break_drain + T_contention_other)|` avec **residual < 3 ms/token sur
  chacun des deux runs**, sinon INVALID.
- INVALID ⇒ rerun à l'identique autorisé, sans changement de code ni de
  paramètres. INVALID n'est ni PASS ni FAIL : c'est un défaut d'instrumentation.

## Critères (inchangés, compteurs exacts, AUCUN epsilon)

- **C1** : pire tok/s_EVAL des deux runs ≥ 2,0 sur CHACUN de G5–G8.
- **C2** : C_engine ≤ C_static sur EVAL pour CHACUN des quatre.
- **C3** : C_engine < C_static strictement sur AU MOINS DEUX holdouts.
- **PASS** = C1 ∧ C2 ∧ C3 ; sinon **FAIL** ; appliqué mécaniquement, y compris
  si la cause est une lenteur spécifique aux workloads neufs.

## Métriques de recherche (publiées, non bloquantes)

Décomposition complète par run, expert_ready, hit rates CAL vs EVAL, capture
vs Bélády (objectif recherche 0,40), U_temporal, comparaison à la référence
dev 12,6 tok/s.

## Clauses

- Moteur/traceur/analyseur : versions hashées du gel, zéro changement.
- Prompt hors gabarit (n_ctx dépassé, EOS précoce sous quota) : ajusté AVANT
  tout run moteur du holdout concerné, annoté dans HASHES (précédent v2).
- Un PASS ne valide ni K3-class ni d'autres modèles ni le format final :
  LE moteur gelé, sur LE véhicule Gemma-4-A4B, en régime holdout.
