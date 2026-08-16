# P1a v0 — Premier run du pipeline réel (2026-08-16)

Chaîne complète avec de vrais octets : NVMe direct I/O → SHA-256 (CNG) → RAM →
H2D → slot-arena VRAM → FFN MoE GPU (mul_mat_id, chemin gate_up fusionné) →
éviction SLRU. Policy P1 gelée (pin 75 % calibré + SLRU 25 %), allocation fixe
par couche, **v0 = chemin critique 100 % synchrone (borne basse honnête)**.

Véhicule : Gemma-4-A4B Q4_0 (30 couches MoE, 128 experts top-8, slab 3,35 Mo).
Pilotage : trace brûlée W5 (calibration 256 tokens, mesure 256 tokens).

## Résultats

| Métrique | Valeur |
|---|---|
| **tok/s réel (synchrone)** | **10,72** — 5,4× le plancher de viabilité |
| ms/token | 93,3 = io 28,2 + **sha 35,7** + h2d 13,1 + gpu 16,0 + policy 0,09 |
| Hit rate SLRU live | 0,888 (cohérent avec la sim : 0,861-0,877) |
| Cold Mo/token | 90,0 (sim prédisait ~110 : ✓ ordre de grandeur) |
| Débit I/O effectif | 3,19 Go/s (lectures bloquantes unitaires, QD1 de fait) |
| Débit SHA effectif | 2,52 Go/s (1 thread — exactement le chiffre P0.2) |
| Débit H2D effectif | 6,89 Go/s (cohérent P0.0) |
| VRAM arenas | 3 212 Mo (32 slots × 30 couches) |
| Boot (préchargement 24×30 experts pinnés) | 2,0 s |

## Lecture

1. **La marge survit au réel.** La question de GPT (« l'implémentation
   conserve-t-elle la marge quand toutes les contraintes physiques sont
   réunies ? ») reçoit sa première réponse : oui, ×5,4 sans aucun overlap.
2. **Le goulot v0 est le SHA mono-thread (38 % du temps)** — exactement ce que
   P0.2 prédisait : 2,5 Go/s/thread face à 90 Mo/token. À 3 threads pipelinés,
   l'étage descend de 35,7 à ~12 ms.
3. Projection pipeline async (overlap complet, étages P0-mesurés) :
   chemin critique ≈ max(io@QD4 ~13,6 ; sha@3t ~12 ; h2d 13,1 ; gpu 16,0)
   ≈ **16 ms/token ⇒ ~60 tok/s** de potentiel — c'est l'étage GPU qui
   deviendrait le goulot, c'est-à-dire exactement là où un moteur d'inférence
   veut être. (Le gpu 16 ms inclut 30 lancements de graphe/token ; CUDA graphs
   et fusion réduiront encore.)
4. Le contrat d'éviction est implémenté : experts du token courant
   non-évincables (préfiguration IoTicket), pas de slot leak (fix demotion).

## P1a.1 — pipeline async (2026-08-16, même trace, 6 workers)

Correctif préalable : bug `std::move` → le premier run async ne synchronisait
pas les fetches avant le FFN (97 tok/s invalides, attente 0.0 = signal du bug).
Chiffres corrigés :

| Mode | ms/token | tok/s | Décomposition |
|---|---|---|---|
| Sync v0 (rappel) | 93,3 | 10,7 | somme des étages |
| **Async, dépendance routeur réelle** | **67,3** | **14,9** | attente fetch 55,5 + gpu 11,3 ; overlap 0,38 |
| **Async + prefetch parfait (borne haute)** | **21,2** | **47,2** | attente 11,7 + gpu 9,2 ; overlap 0,83 |

**La découverte : le *routing-dependency stall* est mesuré à ~46 ms/token**
(67,3 vs 21,2) — c'est LE goulot du moteur réel, pas l'I/O brute ni le SHA.
Terminologie (correctif GPT) : ce ne sont pas 46 ms de « synchronisation du
routeur » — le routeur lui-même coûtera vraisemblablement <1 ms. C'est la
**dépendance causale** routeur L → top-k → misses → fetch → FFN L qui expose
~46 ms de latence I/O autrement masquable. P1a.2 devra décomposer :
router_compute / host_visibility (graph break) / routing-dependent stall / FFN.
Avec la dépendance réelle, le fetch d'une couche ne peut recouvrir que le
minuscule FFN de la couche courante (0,3 ms) : la latence par miss (~2 ms ×
~18 couches à miss) s'additionne presque intégralement. Le prefetch parfait la
masque à 83 %.

Conséquences :
1. **La prédiction de routage revient au premier plan, mais pour la LATENCE,
   pas pour les octets** — exactement la distinction stall_bytes/total_bytes de
   l'amendement v2.1. Un bigramme à ~55 % de précision pourrait masquer environ
   la moitié des 46 ms. À re-chiffrer en mode « prefetch spéculatif + correction ».
2. Réduire la latence par miss est le second levier : lectures chunkées
   (io/sha/h2d pipelinés par morceaux de slab), SHA incrémental, QD par miss.
3. Contention mesurée à 6 lecteurs bloquants concurrents : débit I/O effectif
   descend à ~2 Go/s (vs 6,6 QD structuré P0.1) — le StreamingEngine final doit
   utiliser une vraie file QD plutôt que N lecteurs indépendants.

## P1a.1-préd — le bigramme spéculatif échoue, et on sait précisément pourquoi

Protocole GPT (3 prédicteurs) exécuté sur la trace brûlée :

| Prédicteur | ms/token | tok/s | R = récupération de la fraction oracle |
|---|---|---|---|
| P0 aucun | 67,3 | 14,9 | 0 (référence) |
| **P1 bigramme causal (budget 4, cap n_dyn/2)** | **123,8** | **8,1** | **négatif** — pire que rien |
| P2 oracle | 21,2 | 47,2 | 1 |

Métriques d'échec (celles demandées) : précision **0,095**, recall des miss
0,278, deadline hits 28 % des utiles, **gaspillage 373 Mo/token** (4,1× les
octets utiles !) qui sature le NVMe (io busy 314 ms/token cumulés) et affame
les fetches à la demande — le stall passe de 55 à 99 ms/token.

**La cause structurelle, plus intéressante que l'échec** : en replay, le
bigramme affichait ~0,55 de précision sur le *routage*. Mais ses bonnes
prédictions sont précisément les experts que le cache tient déjà (le hotset).
Une fois les résidents exclus, il ne reste que la queue imprévisible : **le
prédicteur doit prédire les MISS, pas le routage — et conditionné au miss, le
routage de ce modèle est quasi imprévisible.** C'est la troisième défaite
indépendante de la prédiction (P0.3 octets, P0.5 EWMA, P1a.1 latence), toutes
par des mécanismes différents. Le faisceau est complet.

Voies de récupération du stall restantes, par ordre de promesse :
1. **Latence par miss** : file QD structurée (2 Go/s → 6,6 mesurés possibles),
   fetch chunké io/sha/h2d pipeliné (~2 ms → ~1 ms/miss), SHA multi-thread par slab.
   Gain borné mais sûr : stall ~55 → ~25-30 ms/token estimés.
2. Variante prefetch « haute confiance seulement » (seuil de score, budget 1-2) :
   à tester une fois, mais l'espérance est faible vu la précision conditionnelle.
3. Accepter le plancher : 14,9 tok/s réels = 7,4× la cible produit, déjà.

## P1a.1-QD — file I/O structurée (forme StreamingEngine §3.1)

Remplacement des N lecteurs bloquants par : 1 thread I/O possédant la file
(overlapped, QD 8) → pool SHA borné (3) → H2D. Buffers possédés par leur tâche
jusqu'à complétion (contrat IoTicket implémenté).

| Mode | avant (6 lecteurs) | après (QD structurée) |
|---|---|---|
| Dépendance réelle | 14,9 tok/s (67,3 ms) | **15,2 tok/s (65,7 ms)** |
| Oracle | 47,2 tok/s (21,2 ms) | 36,7 tok/s (27,3 ms)* |

\* QD8 + 3 SHA sous-performe les 6 lecteurs×2 (QD effective ~12) en régime
saturé — tunable (QD16), non prioritaire.

**Conclusion affinée, et c'est la vraie valeur de cette mesure : le stall en
dépendance réelle est borné par la LATENCE par miss (~2 ms), pas par la bande
passante.** Avec 1-2 miss en vol, la profondeur de file ne change rien. La
structure en file reste la bonne architecture (elle gagne en régime concurrent
et implémente le contrat de possession des buffers), mais le levier suivant
est la réduction de latence unitaire : **le chunking (read → SHA incrémental →
H2D pipelinés par morceaux de slab), initialement 2e priorité, est maintenant
promu par la mesure** — per-miss ~2 → ~1 ms attendu, stall ~54 → ~27 ms/token,
soit ~30-35 tok/s en dépendance réelle.

État des tok/s réels au fil de P1a : 10,7 (sync) → 14,9 (async) → 15,2 (QD).
Plancher produit : 2. La marge est ×7,6 avant toute optimisation de latence.

## P1a.1-chunk — fetch chunké, sweep de taille (dépendance réelle, 3 workers)

Pipeline intra-tâche : read-ahead K=3 buffers préalloués (overlapped), SHA-256
incrémental en ordre, H2D par chunk après son hash. **Fail-closed préservé** :
le FFN n'est libéré qu'après vérification du digest complet du slab, même si
les octets sont déjà en VRAM.

| Chunk | tok/s | expert_ready/miss | io-wait | sha | h2d |
|---|---|---|---|---|---|
| slab entier | 18,9 | 2,73 ms | 0,83 | 1,37 | 0,42 |
| 1 MiB | 21,2 | 2,46 | 0,43 | 1,40 | 0,50 |
| **512 KiB** | **21,8** | **2,39** | **0,28** | 1,45 | 0,50 |
| 256 KiB | 21,3 | 2,48 | 0,26 | 1,43 | 0,60 |
| 128 KiB | 17,5 | 3,08 | 0,33 | 1,52 | 0,90 |

(La baseline « slab entier » du nouveau pool fait déjà 18,9 vs 15,2 avant :
la lecture de la région down se recouvre avec le hash de gate_up. Le coude
prédit existe : 128 KiB s'effondre sous le bookkeeping/H2D par chunk.)

**Lecture : le chunking a éliminé l'I/O du chemin critique** (io-wait 0,83 →
0,28 ms — les lectures se cachent derrière le hash) **et expose le nouveau
plancher : le SHA séquentiel, ~1,4 ms/miss, incompressible en l'état** — un
digest SHA-256 unique par slab se calcule séquentiellement par nature, le
chunking ne peut pas le paralléliser. GPT avait raison de ne pas graver le
« ~1 ms/miss » : on mesure 2,39 ms.

**Conséquence de format, pas d'implémentation** : pour descendre sous ~1,4 ms,
il faut des **digests par chunk dans `model.ews.idx`** (l'option racine de
Merkle du §8, déjà prévue !) — chunks vérifiables indépendamment ⇒ SHA
parallélisable entre workers ⇒ sha/miss ÷3-4 ⇒ expert_ready ~1,2 ms ⇒
~28-32 tok/s attendus. La mesure vient de transformer une option de design en
exigence chiffrée.

Trajectoire tok/s réels (dépendance causale honnête, plancher 2) :
**10,7 sync → 14,9 async → 15,2 QD → 21,8 chunké (512 KiB)** — marge ×10,9.

## Note de calibration sim ↔ réel (demande GPT)

90 Mo/token réel vs ~110 simulé : même fenêtre (tokens 256-511), même
définition du cold (slab complet par miss), warmup exclu des deux côtés. La
différence vient de la gestion du débordement de probation : le simulateur
Python évince dès que probation > n_prob (capacité dynamique effective réduite),
le C++ n'évince qu'à saturation des slots (les 8 slots dynamiques restent tous
utiles). Le C++ est strictement meilleur et légitime ; hit 0,888 vs 0,861 sim,
d'où ~20 % de cold en moins. À aligner dans le simulateur si on veut une
prédiction exacte, sans impact sur les verdicts rendus (l'écart est dans le
sens conservateur).

## Prochaines étapes P1a

- Pipeline async : pool I/O (QD 4-8), 3-4 threads SHA, double buffering H2D
  sur stream dédié, recouvrement avec le compute de la couche précédente.
- Puis branchement sur la vraie inférence (activations réelles au lieu du
  pilotage par trace) — la synchro top-k du routeur devient mesurable.
- G5-G8 : réservés au verdict end-to-end (gate C+ de P1-GATE.md), après le
  pipeline async.
