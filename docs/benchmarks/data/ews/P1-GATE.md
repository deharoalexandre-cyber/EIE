# Gate P1 (« C+ ») et cadrage P1a/P1b — arbitrage GPT du 2026-08-16

## Position méthodologique, écrite comme telle

Le verdict v3 (STOP & REDESIGN) reste intouché et correct **pour la question
que v3 posait**. P0.5 a montré que cette question — capture ≥ 0,5 du potentiel
Bélády — mesurait la distance à la clairvoyance plus que la qualité du moteur :
neuf politiques raisonnables plafonnent à ~0,40–0,48, deux familles de
prédicteurs échouent, et l'écart résiduel est de la prévoyance pure. Le gate
suivant est donc re-fondé sur la physique. C'est une évolution méthodologique
assumée, décidée AVANT la fabrication de tout nouveau holdout.

## Gate P1 (verdict end-to-end, sur les futurs holdouts G5–G8)

1. **Viabilité I/O** : ≥ 2 tok/s sur chaque holdout (plancher inchangé).
2. **Non-régression** : C_policy ≤ C_static sur chaque holdout (la policy
   dynamique ne doit jamais faire pire que le simple pinning calibré).
3. **Apport dynamique démontré** : amélioration mesurable de C_total vs static
   sur plusieurs workloads (sans normalisation obligatoire par Bélády).

**Métriques de recherche** (publiées, non bloquantes) : capture vs Bélády
(objectif de recherche pré-enregistré : 0,40), U_temporal, cold bytes économisés
vs static, churn, hit rates, distributions par couche.

**Les holdouts G5–G8 sont réservés au verdict du MOTEUR RÉEL** (P1a), pas à un
troisième verdict simulé. G1–G4 restent morts ; les diagnostics restent dev.

## Policy P1 gelée

**Hotset pinné 75 % du budget (calibré) + SLRU sur les 25 % restants.**
Simple, déterministe, explicable, sans prédicteur fragile ; gains mesurés en
P0.5 ; contrôle négatif propre. RoutingPredictor : hors P1 (v2 sur gain démontré
en stall bytes uniquement).

## Séquence P1

- **P1a — pipeline réel, allocation fixe par couche** : prouver la chaîne
  complète NVMe → SHA-256 → RAM → H2D → slot-arena → routeur → FFN → éviction,
  et mesurer ce qui compte : tok/s réel, stall ms/token, cold Mo/token,
  H2D Go/s, profondeur de file, hit rate, VRAM réelle, surcoût SHA, surcoût
  synchro routeur.
- **P1b — allocateur global inter-couches** : retenu dans l'architecture
  (le budget VRAM réel est global ; +0,08 de capture mesuré sur Gemma ; rôle
  naturel du VramManager EIE), mais APRÈS que le moteur fonctionne — on ne
  mélange pas deux nouveautés.

## Contrat de sécurité et de robustesse reconduits en P1a

- SHA-256 par slab à chaque lecture froide, fail-closed (§8 design) —
  3–4 threads au débit pic, pipeliné.
- Durée de vie `AlignedBuf` possédée par `IoTicket` : libération interdite
  avant complétion drainée ou cancel confirmé (leçon DMA P0.1).
