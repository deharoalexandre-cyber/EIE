# Amendement C+ v1.1 — Incident P1-C+-CTX-01 (2026-08-16)

## Constat (historique, non réécrit)

**Protocole C+ original : G5/G8 = INVALID**, déclarés avant tout résultat EVAL
exploitable : leur longueur dépassait le `n_ctx = 4096` compilé dans le candidat
gelé (`ENGINE-FREEZE.md`). Les runs ont échoué sans produire de decode valide ni
exposer de routage EVAL. Ce résultat reste dans l'historique.

Attribution double (arbitrage GPT) : le protocole a demandé ce que le candidat
ne pouvait pas fournir (défaut de spécification) ; ET la limite n_ctx=4096 est
une **limitation fonctionnelle réelle du candidat**, conservée au rapport final.

## Amendement

Motivé exclusivement par cette contrainte connue et mesurable. **Aucune
modification du moteur, de la policy, des seuils, de F, des chunks ou des
critères C1/C2/C3. G6/G7 restent acquis** (runs valides, critères antérieurs
aux données, aucune influence possible sur les seuils).

G5/G8 sont remplacés par **G5'/G8'**, mêmes domaines fonctionnels, contraints
au contexte effectivement supporté par le candidat.

**Portée : un éventuel PASS est explicitement borné à n_ctx ≤ 4096.** Il ne
permet AUCUNE conclusion sur le contexte long (~8k) ; le plafond 4096 est une
limitation documentée du candidat gelé.

## Règles de fabrication déterministes (écrites et hashées AVANT les contenus)

- **G5'** (classe : code C++, matériau ORIGINAL de G5 conservé, troncature
  mécanique) : source = `p1a_pipeline/main.cpp[0:8500]` (préfixe strict du
  matériau G5 original), même gabarit de prompt Gemma que G5, n_predict 512.
  Contrôle au moment de la trace CAL : `n_prompt + 512 + 64 ≤ 4096`, sinon
  réduction mécanique par pas de 500 caractères jusqu'à satisfaction, chaque
  réduction annotée.
- **G8'** (classe re-déclarée : *contexte maximal supporté par le candidat
  gelé, sans dépasser n_ctx=4096, marge réservée au décodage*) : source =
  `llama-context.cpp[0:9200]` (préfixe strict du matériau G8 original), même
  gabarit de prompt, n_predict 256. Contrôle : `n_prompt + 256 + 64 ≤ 4096`,
  même règle de réduction par pas de 500.
- CAL = première moitié du decode (256 / 128), EVAL = seconde ; toutes les
  autres dispositions du protocole hashé s'appliquent inchangées.

## Séquence

amendement (ce fichier) → hash → création G5'/G8' → hash → traces CAL →
hotsets hashés → runs (mode0, R1, R2, R3, no_wait) → traces EVAL des QUATRE
holdouts (G5', G6, G7, G8') → cstatic/cengine → verdict mécanique C1∧C2∧C3.
