# P0.3 — Seuil de kill v2 (figé AVANT lecture des résultats)

**v1 rédigée le 2026-08-16 avant toute trace. v2 amendée le 2026-08-16, toujours
avant la première trace de mesure, sur audit GPT (métrique physique primaire,
hard floor produit) et validation Fable (budget par couche, courbe complète,
seeds, GO explicite, second point de régime).**
**SHA-256 de ce fichier et de `replay.py` consignés dans `HASHES.txt` avant le
premier token tracé.** Toute modification postérieure sera annotée comme telle.

## Cadre de mesure

- Modèle principal : Mixtral 8x7B Instruct Q4_K_M — 32 couches MoE, 8 experts/couche,
  top-2, slab = 114,4 Mo/expert. Coût plein sans cache : `C0 = 7,32 Go/token`.
- **Second point de régime (diagnostic, hors verdict)** : Gemma 4 26B-A4B Q4_0 —
  128 experts/couche, top-8, slab ≈ 3,35 Mo (gate_up 2,23 + down 1,11). Une trace
  (W2 prose) pour tester le transfert du régime grossier vers le régime fin.
- 4 workloads Mixtral, ≥ 512 tokens décodés (+ prefill) : W1 code C++, W2 prose
  française, W3 sondes NMM (**proxy**, à re-tracer avec les vraies sondes), W4
  contexte long ~8k.
- Verdict calculé sur le **segment decode** ; stats prefill informatives.
- **Budget PAR COUCHE** (confirmé) : slots_par_couche = F × n_expert. Les distances
  de réutilisation ne se mélangent pas entre couches ; un budget global laisserait
  des couches affamées masquer des couches riches.
- Courbe complète obligatoire à F ∈ {12,5 % ; 25 % ; 37,5 % ; 50 % ; 65 %} pour
  l'oracle ET les politiques pratiques. **Le verdict se lit à F = 25 %** (point
  pré-enregistré) ; si le coude est ailleurs, c'est une donnée de design, pas un échec.
- Prefetch facturé : tout octet préfetché est compté au moment du fetch, utile ou non.
- Reproductibilité : decode greedy argmax (déterministe, pas de seed d'échantillonnage),
  `n_gpu_layers`, `n_ctx`, arch et comptes d'experts loggés dans l'en-tête de chaque trace.

## Métrique primaire : octets froids par token

Pour chaque politique P, `C_P` = octets lus (NVMe simulé) par token décodé, et
`R_P = 1 − C_P / C0`. Politiques simulées, par couche :

| Sigle | Politique |
|---|---|
| C0 | sans cache (référence physique) |
| C_static | hotset statique : top-fréquence appris sur 1re moitié, évalué sur 2de |
| C_online | top-fréquence apprise en ligne (cumul) |
| C_LRU | LRU |
| C_LRUK | LRU-K (K=2) |
| C_pred | LRU-K + prefetch bigramme L→L+1 budgeté (raté facturé) |
| C_Belady | oracle clairvoyant — borne physique de la localité |

La non-uniformité du routage (experts chauds) est une propriété QUE le design
exploite : elle appartient au gain d'EWS, elle n'est pas soustraite. C'est le
correctif central v1→v2 : l'uplift vs budget aveugle sous-évaluait le hotset.

## Métrique diagnostique : localité temporelle

`U_temporal = (H_Belady − H_static) / (1 − H_static)`

Faible avec H_static élevé ⇒ le gain vient du biais de popularité ⇒ EWS se
simplifie (pinning calibré), il ne meurt pas.

## Seuils (verdict à F = 25 %, sur les 4 workloads Mixtral)

**S1 — hard kill, lié à l'objectif produit (§9 : Mixtral > 2 tok/s).**
`tok/s_Belady = 4,5 Go/s ÷ C_Belady`. Si `tok/s_Belady < 2,0` sur **les 4
workloads** : même un cache clairvoyant reste sous le plancher produit avant
calcul, SHA et décompression. La prémisse v1 est morte, on documente et on stoppe.

**S2 — stop & redesign (le potentiel existe, la politique v1 ne le capture pas).**
`capture = (C_static − C_v1) / (C_static − C_Belady)`, avec C_v1 = min(C_LRUK, C_pred).
Si S1 passe mais `capture < 0,5` sur les 4 workloads : redesign de la policy
avant P1, pas de kill. Garde-fou : si `C_static − C_Belady < 2 % de C0`, la part
dynamique est négligeable ⇒ capture := 1 par convention (rien à capturer, voir GO-simplifié).

**GO — les trois sorties écrites du document.**
- **GO-simplifié** : S1 et S2 passent, et `U_temporal < 0,10` avec H_static
  portant l'essentiel du gain ⇒ P1 sans cache dynamique complexe : pinning
  calibré + éviction simple. C'est un succès, pas un lot de consolation.
- **GO-dynamique** : S1 et S2 passent, `U_temporal ≥ 0,10` ⇒ P1 tel que conçu
  (LRU-K, prefetch selon règle prédicteur).
- **KILL / REDESIGN** : S1 ou S2 déclenché, voir ci-dessus.

**Règle prédicteur (décision sur le coût physique, pas la précision).**
Amendement v2.1 (2026-08-16, toujours AVANT la première trace) : en octets
totaux, un prefetch ne peut jamais réduire `C` — un expert préfetché utile
aurait été lu de toute façon au miss, et les ratés s'ajoutent ; le critère
« C_pred ≤ 0,95 × C_LRUK » serait insatisfaisable par construction. Ce que le
prefetch améliore physiquement, ce sont les octets **sur le chemin critique** :
`C_stall` = octets des miss à la demande restants (non couverts par prefetch),
c'est eux qui bloquent le token. Règle : le bigramme entre en v1 ssi
`C_stall ≤ 0,95 × C_LRUK` **et** `C_pred_total ≤ 1,2 × C_LRUK` (le gaspillage
de bande passante reste borné). Les deux chiffres et précision/rappel publiés.

**S3 — publiés quel que soit le verdict** : courbes C_P(F) complètes par workload,
distributions par couche, distances de réutilisation, `tok/s_max = 4,5 ÷ C_P`
pour chaque politique, comparaison Mixtral vs Gemma-4-A4B (régime grossier vs fin).

## Périmètre honnête

Un GO Mixtral ne valide pas la classe K3 (896 experts, top-16). La trace
Gemma-4-A4B (128/top-8) donne un second point de régime et rend le verdict
partiellement transférable ; K3 reste un point de mesure P2 sur vrai checkpoint.
Un S1 déclenché sur les 4 workloads Mixtral tue la prémisse v1 quel que soit
le régime fin.
