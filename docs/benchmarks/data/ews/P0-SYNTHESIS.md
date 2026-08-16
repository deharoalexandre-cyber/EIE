# EWS P0 — Synthèse de dérisquage (2026-08-16)

Protocole : `p03_routing_traces/KILL-THRESHOLD.md` v2, hashé avant le premier
token tracé (`HASHES.txt`). Résultats détaillés : `p03_routing_traces/RESULTS.md`
(+ `.json`), `p01_nvme_io/results.txt`, `p02_sha_dequant/results.txt`.

## Verdict formel, tel que pré-enregistré

**S1 (hard kill) : DÉCLENCHÉ sur les 4 workloads Mixtral.**
À F = 25 %, même le cache clairvoyant (Bélády) donne 1,04–1,10 tok/s @ 4,5 Go/s,
sous le plancher produit de 2 tok/s. Conformément au protocole : pas de P1 sur
cette prémisse, on documente pourquoi — voir ci-dessous, la cause est physique
et instructive.

S2 : non déclenché (mais capture ≤ 0 partout : LRU-K ne bat même pas le hotset
statique sur Mixtral). Règle prédicteur : non satisfaite sur Mixtral (0/4).

## Pourquoi : Mixtral n'a pas de localité à exploiter

- Entropie de routage agrégée : **2,997–2,999 bits sur 3,0 max** sur les quatre
  workloads. Le load-balancing de Mixtral fait exactement son travail : le
  routage est quasi uniforme, il n'y a pas d'experts chauds.
- H_static(25 %) ≈ 0,29–0,33 contre 0,25 trivial ; Bélády 0,40–0,43. Le peu de
  gain vient de la localité temporelle courte (p50 de réutilisation = 2 tokens),
  pas d'un hotset.
- Avec 8 experts × slab 114 Mo, chaque miss coûte trop cher : C_Bélády(25 %) ≈
  4,1–4,3 Go/token. La physique ne suit pas, quel que soit le cache.

## Mais le régime cible d'EWS raconte l'inverse (trace diagnostique Gemma-4-A4B)

128 experts top-8, slab 3,35 Mo — le régime structurel des cibles réelles
(K3-class : 896 experts top-16) :

| F | H_static | H_LRU-K | C_LRU-K (Go/tok) | tok/s LRU-K @4,5 Go/s |
|---|---|---|---|---|
| 12,5 % | 0,698 | 0,651 | 0,27 | 16,4 |
| 25 % | 0,865 | 0,861 | 0,11 | 41,1 |
| 37,5 % | 0,948 | 0,942 | 0,05 | 98,7 |

Un hotset de 16 experts sur 128 capte **70 %** des routages ; 32 en captent 86 %.
La localité que Mixtral n'a pas, le MoE fin l'a massivement. Et P0.1 mesure
**6,6 Go/s** réels sur des slabs de 3,35 Mo (QD≥4) contre 3,3 Go/s sur les slabs
de 114 Mo : le régime fin est doublement avantagé (localité ET débit disque).
Aux débits mesurés, le **plafond I/O** de Gemma-A4B à F=25 % dépasse 60 tok/s
(6,6 Go/s ÷ 0,11 Go/token). C'est une borne supérieure imposée par le disque,
pas un débit attendu : l'inférence réelle sera limitée par le compute, les
synchros et l'upload H→D. L'important est la marge : ×30 par rapport au
plancher produit de 2 tok/s.

## Chiffres physiques (P0.1 / P0.2, machine RTX 4090 Laptop, NVMe Gen4)

- NVMe direct (NO_BUFFERING, overlapped) : pic **6,7 Go/s** (16 Mo, QD8) ;
  3,35 Mo → 6,6 Go/s dès QD4 ; 114 Mo → ~3,3 Go/s ; QD1 pénalise tout (×0,4–0,7).
  Le chiffre io_uring Linux reste à reprendre sur la machine cible.
- SHA-256 (CNG/SHA-NI) : **2,47 Go/s/thread**, scaling linéaire (35 Go/s à 16
  threads). Au plancher protocole (4,5 Go/s), 2 threads suffisent ; au pic NVMe
  mesuré (6,6–6,7 Go/s), il en faut 3, et 3–4 pour de la marge. Le SHA par slab
  n'est pas « gratuit » mais **suffisamment rapide pour être pipeliné sans
  devenir le goulot** — c'est la formulation défendable, et elle confirme §8.
- Déquant (octets compressés consommés, par thread) : q4_K 2,3 Go/s, q4_0 1,7,
  q6_K 1,2. Quelques threads suffisent ; et v1 streame les quants GGUF tels
  quels (pas de décompression sur le chemin), donc hors chemin critique.

## Données S3 à verser au dossier (pré-enregistrées comme publiables)

- À F = 65 % — le budget effectif du scénario §9 « Mixtral sur 24 Go » —
  LRU-K donne 2,05–2,09 tok/s @ 4,5 Go/s : la cible produit originelle passe
  de justesse à son budget de déploiement réel. Le point de verdict F = 25 %
  était volontairement plus discriminant ; les deux lectures sont publiées.
- Bigramme : précision ≈ 0,45 (Mixtral), 0,58 (Gemma) ; n'améliore les stall
  bytes nulle part sauf Gemma à F = 12,5 %. La prédiction L→L+1 n'est PAS le
  levier dominant ; le hotset l'est.
- Trouvaille sécurité (P0.0-bis) : corruption 1 octet d'un slab q4_K absorbée
  par la quantisation q8_K des activations sur certains combos ⇒ l'intégrité
  des poids ne peut pas être déduite du comportement ⇒ SHA-256 par slab
  obligatoire, indépendamment de toute détection comportementale (EAS).

## Lecture d'ensemble proposée (décision = Alex/Fable/GPT)

P0 ne tue pas EWS ; il tue **Mixtral comme véhicule et comme cible v1**. Les
deux hypothèses du design §2.1 se départagent expérimentalement : la
non-uniformité du routage est fausse sur MoE grossier load-balancé, massive sur
MoE fin ; la localité temporelle existe partout mais elle est secondaire. Si le
périmètre v1 est re-scopé sur un MoE fin (Gemma-4-A4B local, Qwen3-30B-A3B,
trajectoire K3-class), tous les voyants P0 sont au vert : slot-arena démontrée
(P0.0), localité démontrée (P0.3-Gemma), I/O et SHA dimensionnés (P0.1/P0.2)
avec un avantage structurel aux petits slabs. Mixtral reste utile comme
contrôle négatif dans les benchs publiés — publier ce résultat négatif renforce
la crédibilité du reste.

Re-scoper §9 est une modification de design, pas un contournement du seuil :
le kill S1 tel qu'écrit reste déclenché et documenté ici.
