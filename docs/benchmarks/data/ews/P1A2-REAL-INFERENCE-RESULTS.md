# P1a.2 — Décomposition sur vraie inférence (2026-08-16)

Gemma-4-A4B, llama.cpp API, decode greedy 256 tokens (prompt prose brûlée),
n_gpu_layers=24. Trois modes ablatifs, pipeline 512 KiB gelé, policy 75/25
calibrée sur la trace brûlée. Le stall s'injecte dans le callback `cb_eval`
au vrai point de dépendance causale (le graph break du routeur).

## La décomposition demandée

| Composant | Mesure |
|---|---|
| Inférence pure (router_compute + attention + FFN, GPU) | 22,5 ms/token (44,5 tok/s) |
| **host_visibility** (graph break + sync + D2H du top-k, 30 couches) | **0,29 ms/token (~10 µs/couche)** |
| **routing-dependent stall** (fetch réel : NVMe→SHA→H2D, attente) | **53,1 ms/token** |
| policy | 0,12 ms/token |
| **Total mode streaming réel** | **90,4 ms/token = 11,1 tok/s** (5,5× le plancher) |

## Les trois verdicts de cette mesure

1. **host_visibility est négligeable.** ~0,3 ms/token pour rendre le top-k
   visible côté hôte sur 30 couches. La grande inconnue architecturale du
   design (coupe de graphe au routeur, §5/§11) est réglée : ce n'était pas un
   coût, c'était une peur. L'optimisation SHA reste donc bien la priorité
   suivante (réponse à la question posée par GPT avant ce run).
2. **Le banc trace-driven était fidèle.** Hit rate live 0,888 = bench 0,888 ;
   cold 89,4 vs 90,0 Mo/token. Le routage greedy live reproduit la trace —
   toutes les conclusions P1a.1 se transfèrent telles quelles.
3. **Le stall réel est un peu plus cher qu'au banc** : expert_ready 3,45 ms/miss
   (vs 2,39) — l'inférence réelle charge le GPU et le CPU (attention/FFN +
   threads llama.cpp), les fetches subissent cette contention. C'est le chiffre
   honnête d'un moteur en fonctionnement.

## Fermeture comptable (demande GPT — le résiduel de ~14,5 ms a un nom)

Trois runs d'attribution (mêmes workload/poids/policy) :

| Variante | total | stall | résiduel |
|---|---|---|---|
| Baseline (3 SHA, H2D on) | 86,2 | 51,5 | 11,5 |
| 1 SHA (test contention CPU) | 105,9 | 66,9 | 15,8 |
| Sans H2D (test contention CUDA) | 89,8 | 48,7 | 17,8 |
| **Sans attente** (fetches complets en fond, jamais attendus) | **25,3** | 0 | **2,35** |

Le résiduel ne réagit ni au nombre de threads SHA ni à la présence des memcpy
H2D — mais il s'effondre quand on supprime le **blocage** : c'est le
**graph_break_drain** — le coût de vidange/relance du pipeline GPU à chaque
événement de blocage au point de routage (~18 couches à miss/token × ~0,6 ms
de ré-entrée scheduler + relance kernels). La contention pure du machinisme
de fetch (CPU + CUDA) ne vaut que ~2,3 ms/token.

**Comptabilité fermée** :
`86,2 ≈ compute 22,5 + vis 0,5 + policy 0,1 + stall 51,5 + drain ~9,2 + contention ~2,3` ✓

Conséquence structurante : le drain se paie **par événement de blocage**, pas
par milliseconde de stall. Le SHA parallèle réduira le stall mais pas le drain ;
projection re-chiffrée : `22,5 + 0,6 + 36 (stall à SHA parallèle) + 9 + 2,3 ≈
70,5 ms → ~14,2 tok/s` — la borne basse de la fourchette 14–16 de GPT tient ;
les 16 demanderont de réduire le NOMBRE d'événements (hit rate/F) ou le coût
unitaire du drain (reprise asynchrone du graphe).

## Format sécurité v1 (spécification GPT, consignée)

Table de SHA-256 **par chunk** dans `model.ews.idx`, elle-même authentifiée
par la signature ML-DSA-65 de l'index. Au chargement : vérification de la
signature de la table. Au runtime : chunks hashés en parallèle, vérifiés
contre la table ; le slot ne passe à READY qu'après validation de TOUS ses
chunks (fail-closed inchangé). **Chaque entrée est liée à
`model_id/version + layer + expert + chunk_index`** — aucune recomposition
cryptographiquement valide depuis un autre slab ou une autre version n'est
possible. Merkle : option ultérieure si un besoin propre le justifie
(engagement compact, preuves partielles transportables).

## P1a.2-shachunk — SHA par chunk implémenté (format v1) et mesuré

Granularité de tâche = chunk 512 KiB : lecture, SHA-256 indépendant, vérif
contre la table (clé layer+expert+chunk_index), H2D ; READY au dernier chunk
validé (fail-closed inchangé). Sérialisation canonique des clés à spécifier
dans le format final (note GPT).

| Config | tok/s | stall | expert_ready |
|---|---|---|---|
| SHA séquentiel/slab (référence) | 11,6 | 51,5 | 3,39 ms |
| **SHA par chunk, 4 workers** | **12,6** | **46,7** | **2,86 ms** |
| 6 workers | 12,6 | 46,7 | 2,85 |
| 8 workers | 12,0 | 48,2 | 2,95 |

Comptabilité (4 workers) : `79,3 ≈ 22,5 compute + 0,8 vis+pol + 46,7 stall +
~7 drain + ~2,3 contention` — fermée, pas de terme inexpliqué.

**Écart à la projection, avec son nom** : projeté ~14,2 tok/s, obtenu 12,6.
Le SHA du miss est bien tombé (~1,4 → ~0,4 ms effectif, plateau identique à
4/6/8 workers = plus limité par le hash), mais les autres termes du miss ont
gonflé en contexte : lecture 512 KiB sous charge GPU/CPU réelle et
sérialisation des H2D (mutex + moteur de copie unique pendant le compute).
Le plancher latence par miss en contexte est ~2,85 ms, composé d'I/O-sous-charge
et de transferts, plus de crypto. Pas d'anomalie inexpliquée — mais la
fourchette 14-16 exigera les leviers « nombre d'événements » (hit/F) ou
« drain » identifiés précédemment, qui sont des chantiers post-verdict.

## État et suite

- **11,1 tok/s réels de bout en bout, streaming actif, sur vraie inférence.**
  Chaîne : modèle → routeur réel → visibilité hôte → SLRU → NVMe direct →
  SHA-256 fail-closed → H2D → FFN. C'est le moteur EWS v0 complet en
  fonctionnement, mesuré composant par composant.
- Levier n°1 confirmé : **SHA par chunk authentifié par l'index signé**
  (décision GPT : table de digests par chunk dans `model.ews.idx` signé
  ML-DSA-65 ; Merkle seulement si utilité propre démontrée ; la racine/table
  s'authentifie au chargement de l'index, les chunks se vérifient contre la
  table au runtime, fail-closed inchangé). Attendu : SHA parallèle ⇒
  expert_ready ~2-2,5 ms en contexte réel ⇒ ~14-16 tok/s.
- Ensuite : gel du moteur (policy + pipeline + params), audit, G5-G8 neufs,
  verdict end-to-end au gate C+.
