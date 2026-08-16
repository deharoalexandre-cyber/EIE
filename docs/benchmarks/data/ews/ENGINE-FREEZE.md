# EWS P1a — GEL DU CANDIDAT (2026-08-16, confirmé par GPT ✅)

**À partir de ce gel : aucune modification de policy, F, chunk, drain, H2D,
cache ou pipeline avant le verdict C+ sur G5–G8.** Toute retouche invaliderait
le candidat et exigerait un re-gel documenté.

## Le candidat

| Paramètre | Valeur gelée |
|---|---|
| Policy | hotset pinné 75 % (calibré top-freq) + SLRU 25 % (probation/protected ½-½) |
| Budget | F = 25 % (32 slots/couche sur Gemma-4-A4B 128 experts) |
| Fetch | chunks 512 KiB alignés 4 KiB, granularité de tâche = chunk |
| SHA | SHA-256 indépendant par chunk, 4 workers, vérifié contre table |
| Sécurité | fail-closed ; slot READY uniquement après validation de TOUS les chunks ; experts routés du token courant non évincables |
| Visibilité routeur | cb_eval au graph break, lecture top-k (≈10 µs/couche) |
| **Performance de référence (dev, avant holdouts)** | **12,6 tok/s end-to-end réels = 6,3× le plancher** (79,3 ms/token : 22,5 compute + 0,8 vis+pol + 46,7 stall + ~7 drain + ~2,3 contention — comptabilité fermée) |

## Format de la table de digests (sérialisation actuelle du candidat)

Sérialisation BENCH telle que testée (la canonique viendra avec `eie-ews-pack`) :
- Clé : entier 64 bits `(layer << 40) | (expert << 16) | chunk_index`
  (modèle/version implicites par processus — le format final lie explicitement
  `model_id/version + layer + expert + chunk_index` et la table est signée
  ML-DSA-65 ; sérialisation canonique non ambiguë à spécifier, note GPT).
- Valeur : digest SHA-256 (32 octets) du payload du chunk (octets du slab
  couverts par le chunk, hors padding d'alignement).
- Découpe : régions gate_up puis down du slab, grille alignée 4 KiB,
  pas 512 KiB, indices de chunks consécutifs sur les deux régions.

## Manifest SHA-256 du candidat gelé

- `p1a2_real_inference/main.cpp` (moteur) :
  `45db589b636414c7b808f10b07ba99835448bf02b982624332f16ee4220cae62`
- `p1a2_real_inference/CMakeLists.txt` :
  `9f5622f87dcb61db84417751bed433e5eea194e0bfc21d4c29c5a157e374d7b4`
- `p1a2_real_inference/build/Release/ews_p1a2.exe` (binaire testé) :
  `15f43583d6d414f01df23de272cf04b2c4216723ec68101984acc962721f434f`
- `p03_routing_traces/replay.py` (analyseur) :
  `ae039126eaf9bbb9ef1774ed5261cea3ff64642fa02ee925b4bb35126dd565af`
- `P1-GATE.md` (gate C+) :
  `cc64e4e1484f54182a2eb5bd3c8eb5f07c846ca0844e99949103b2d874f8eb43`
- `p03_routing_traces/tracer/main.cpp` (traceur, inchangé depuis v2) :
  `846d0b6b5cbd23ab1e107e6cfa60a30120d392f100da89f0c5750bb681123784`
- Modèle : Gemma-4-A4B Q4_0, SHA `3eca3b8f…a51d` (déjà au manifest v3)
- Runtime : llama.cpp `fae3a28`, MSVC 14.44, driver 595.79 (inchangés)

## Circuit restant

1. **Audit GPT** du candidat gelé + du protocole de verdict (brouillon ci-joint :
   `VERDICT-CPLUS-PROTOCOL-DRAFT.md`)
2. Go Alex
3. Création G5–G8 totalement neufs (aucun matériau réutilisé de W*, G1–G4, Q*)
4. Hash holdouts + protocole finalisé dans HASHES.txt
5. Première exécution → **application mécanique de C+**
