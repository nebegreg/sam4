# Changelog - SAM3 + Depth Anything 3 Roto Ultimate

## v0.4.1 - 2025-11-27 - CORRECTIONS MAJEURES

### 🔧 Corrections critiques

#### SAM3 Backend
- **CORRIGÉ** : Utilisation incorrecte de l'API transformers pour SAM3
  - ❌ Ancien : Importait depuis `transformers` (classes inexistantes)
  - ✅ Nouveau : Utilise l'API officielle du repo GitHub `facebookresearch/sam3`
  - Imports corrects : `sam3.model_builder.build_sam3_image_model`, `sam3.model.sam3_image_processor.Sam3Processor`

- **CORRIGÉ** : Workflow SAM3 complètement réécrit
  - Image PCS : `processor.set_image()` + `processor.set_text_prompt()`
  - Image PVS : `processor.set_image()` + `processor.set_point_prompt()` / `set_box_prompt()`
  - Vidéo : `video_predictor.handle_request()` avec session-based workflow

- **AJOUTÉ** : Gestion temporaire des frames pour le tracking vidéo
  - Les frames PIL.Image sont sauvegardées dans un répertoire temporaire
  - Compatible avec l'API SAM3 qui attend des chemins de fichiers ou JPEG folders

#### Depth Anything 3 Backend
- **VÉRIFIÉ** : L'API DA3 était déjà correcte
  - Utilise `depth_anything_3.api.DepthAnything3.from_pretrained()`
  - Appelle `model.inference(paths)` avec liste de chemins
  - Extraction correcte des attributs : `depth`, `conf`, `extrinsics`, `intrinsics`

### 📦 Installation

#### Nouveau script d'installation
- **AJOUTÉ** : `install_models.sh` pour installation automatique
  - Clone et installe SAM3 depuis GitHub
  - Clone et installe Depth Anything 3 depuis GitHub
  - Instructions claires pour téléchargement des checkpoints

#### Requirements mis à jour
- **MODIFIÉ** : `requirements.txt`
  - PyTorch 2.7+ (requis pour SAM3)
  - TorchVision 0.20+
  - Suppression de `transformers` (pas utilisé pour SAM3)
  - Notes sur installation de SAM3 et DA3 depuis GitHub

### 📖 Documentation

#### README complètement réécrit
- Installation détaillée avec prérequis (Python 3.12+, CUDA 12.6+)
- Workflow étape par étape pour tous les modes (PCS/PVS image/vidéo)
- Section Troubleshooting exhaustive
- Notes de performance (RTX 4090)
- Architecture du projet
- Crédits et références vers les repos officiels

### 🔍 Problèmes résolus

1. **Erreur de chargement SAM3** : "ImportError: cannot import name 'Sam3Model' from 'transformers'"
   - Cause : Le code utilisait des imports inexistants depuis transformers
   - Solution : Utilisation de l'API officielle du repo SAM3

2. **Erreur de chargement modèles** : Le modèle ne se chargeait pas correctement
   - Cause : Mauvaise API et mauvais imports
   - Solution : Réécriture complète du backend avec la vraie API

3. **Tracking vidéo non fonctionnel** : Les méthodes de tracking vidéo utilisaient une API inexistante
   - Cause : API fictive basée sur des suppositions
   - Solution : Implémentation correcte avec `handle_request()` et sessions

### 📝 Notes importantes

- **SAM3** nécessite maintenant :
  - Python 3.12+ minimum
  - Clone du repo officiel : `git clone https://github.com/facebookresearch/sam3.git`
  - Installation locale : `pip install -e .` dans le dossier sam3/
  - Authentification HuggingFace pour télécharger les checkpoints

- **Depth Anything 3** nécessite :
  - Clone du repo officiel : `git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git`
  - Installation locale : `pip install -e .` dans le dossier Depth-Anything-3/
  - Installation de xformers : `pip install xformers`

### ⚠️ Breaking Changes

- Le fichier `sam3roto/backend/sam3_backend.py` a été complètement réécrit
- Les signatures des méthodes restent compatibles (pas de changement dans `app.py`)
- Les utilisateurs doivent réinstaller les dépendances avec le nouveau workflow

### 🎯 Prochaines étapes recommandées

1. Exécuter `bash install_models.sh` pour installer SAM3 et DA3
2. Configurer l'authentification HuggingFace : `huggingface-cli login`
3. Tester le chargement de SAM3 avec `facebook/sam3-hiera-large`
4. Tester le chargement de DA3 avec `depth-anything/DA3-LARGE`

---

## v0.4 - Version originale (avec bugs)

- Interface PySide6 fonctionnelle
- Support PCS/PVS pour SAM3 (API incorrecte)
- Support Depth Anything 3 (API correcte)
- Post-processing alpha (matte refinement)
- RGB cleanup (despill, edge extend)
- Exports multiples (PNG, ProRes4444, PLY, Blender)

❌ **Problème majeur** : Le backend SAM3 utilisait une API fictive qui n'existait pas
