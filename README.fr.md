# EIE : Elyne Inference Engine

**Langue : Français · [English](README.md)**

**Un moteur local d'inférence pour les modèles GGUF : conversation, embeddings et exécution configurable de plusieurs modèles.**

EIE est un serveur C++17 publié sous [licence Apache 2.0](LICENSE). Il charge des modèles sur la machine de l'utilisateur et expose une API HTTP dont une partie reprend le format OpenAI. Un seul modèle suffit ; les groupes de modèles sont facultatifs. Le moteur ne fournit pas à lui seul la mémoire, les outils ou l'identité d'une application comme Elyne Next.

## Ce que contient le dépôt

- Le moteur, son serveur HTTP, ses profils de configuration et ses scripts de compilation.
- Des chemins de construction pour Windows, Linux, macOS et Android, dont la qualification varie selon la plateforme.
- Des rapports de mesure et des reçus locaux, avec les limites propres à chaque essai.
- **EWS**, une voie expérimentale qui charge à la demande des poids d'experts pour certains modèles MoE.

Le [README anglais](README.md) décrit l'API, la compilation et la configuration en détail. Le [registre des affirmations](docs/CLAIMS_AUDIT.md) distingue ce qui est implémenté, vérifié localement, rapporté par les mainteneurs ou encore non qualifié. La [feuille de route](docs/ROADMAP_TO_CLAIMS.md) conserve les travaux ouverts.

## Architecture en bref

```mermaid
flowchart LR
    A[Application locale] -->|HTTP : chat / embeddings| S[Serveur EIE]
    S --> M[Gestionnaire de modèles GGUF]
    M --> B[Backend CPU / GPU]
    M -. Profil expérimental .-> E[EWS : cache d'experts]
```

EIE peut traiter les modèles séparément ou appliquer une stratégie de groupe parallèle, séquentielle ou en éventail. Ces chemins sont implémentés, mais leur fiabilité sous charge n'est pas qualifiée à l'échelle d'une flotte. L'API est un **sous-ensemble** du format OpenAI, sans promesse de compatibilité complète.

## Résultats et limites

| Périmètre | État documenté |
| --- | --- |
| Inférence locale Windows/CUDA et service HTTP | Essais bornés avec modèles réels ; [reçu du service](docs/benchmarks/serving-functional-20260908.md). |
| Inférence Windows/HIP sur AMD Radeon 8060S | EIE a servi Gemma 12B, Bonsaï 27B et Nomic sur le GMKtec EVO-X2, toutes leurs couches sur `ROCm0`. Ce build PrismML a été retenu pour les poids ternaires de Bonsaï ; les deux autres modèles ont servi de contrôles. [Mesures et limites](docs/benchmarks/x2-amd-20261123.md). Ce build n'intègre pas EWS ; l'essai GLM/EWS ci-dessous utilise une compilation distincte. Le script Linux/ROCm reste à tester. |
| Elyne Next, modèle résident 12B et auxiliaire 26B via EWS | Aller-retour réel de l'application observé localement ; le résultat ne qualifie pas toute sa durée de vie ni un déploiement à grande échelle. |
| EWS sur Gemma 26B | Consommation effective des poids chargés et coexistence avec Next vérifiées dans les profils publiés ; [rapport](docs/benchmarks/ews-consumed-20260905.md). |
| EWS sur GLM | Après les contrôles numériques et l'aller-retour Next bornés sur portable, une compilation distincte avec Unsloth et EWS a servi le modèle GLM-5.3-Flash de 199,7 Go sur le X2 sous Windows/HIP, **sans PrismML**. Avec 47/47 couches placées sur `ROCm0` et 128 emplacements de cache d'experts, un appel mesuré a atteint **2,103 tokens/s** ; une autre réponse s'est terminée normalement. [Rapport](docs/benchmarks/glm53-ews-hip-x2-20261124.md) · [reçus](docs/benchmarks/data/ews/glm53-ews-hip-x2-20261124.json). Un seul appel par profil, sans référence native appariée ni coexistence avec GRCA. |
| Android arm64 | Bundles CPU `dotprod` et `i8mm` essayés avec Gemma sur Galaxy Z Flip6 ; [reçu et limites](docs/benchmarks/android-neon-zflip6-20260914.md). Ce n'est ni une application Android complète ni une qualification NPU/GPU. |
| Apple Silicon | Fonctionnement et mesures sur MacBook Pro rapportés par les mainteneurs ; [conditions](docs/benchmarks/macos-apple-silicon-20260913.md). Aucune réplication indépendante revendiquée. |

Les anciennes comparaisons générales de vitesse et de VRAM avec Ollama ont été **retirées** : elles ne reposaient pas sur des versions, configurations et mesures brutes appariées. Les résultats EWS d'août et de septembre décrivent des chemins différents et ne doivent pas être fusionnés.

## Pour démarrer

Consultez le [guide Windows](docs/windows.md), le [guide macOS](docs/macos.md), les [composants Android](mobile/README.md) ou les [instructions EWS](docs/ews/runtime-port.md), selon la plateforme. Les poids des modèles et l'application Elyne Next ne sont pas inclus dans ce dépôt.

## Licence et contact

EIE est publié sous [Apache 2.0](LICENSE), avec les notices tierces du dépôt. Pour une intégration ou un échange technique : [Elyne Corp](https://elynecorp.fr) · [admin@elynecorp.fr](mailto:admin@elynecorp.fr).
