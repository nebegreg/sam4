# SAM3 + Depth Anything 3 — Roto Ultimate PRO v0.4 (PySide6)

Application standalone professionnelle pour le rotoscoping et l'estimation de profondeur, conçue pour les pipelines VFX (Autodesk Flame, Nuke, etc.).

## Fonctionnalités

### Segmentation et Tracking (SAM3)
- **PCS (Promptable Concept Segmentation)** : Segmentation par prompts texte ("person", "red dress", "hard hat")
- **PVS (Promptable Visual Segmentation)** : Segmentation interactive avec points +/- et boîtes
- **Tracking vidéo** : Propagation temporelle automatique avec SAM3
- Support des keyframes pour raffinage manuel

### Refinement Alpha
- **Nettoyage** : Fill holes, remove dots, grow/shrink
- **Edge refinement** : Border fix, feather, trimap distance transform
- **Temporal smoothing** : Stabilisation temporelle des masques
- **Edge motion blur** : Flou de mouvement basé sur optical flow (expérimental)

### RGB Cleanup
- **Despill** : 3 modes (Green average, Blue average, Physical auto-BG)
- **Edge extend / Pixel spread** : Extension des bords pour éliminer les halos
- **Luminance restore** : Préservation de la luminosité après despill
- **Premultiply** : Export straight ou premultiplied

### Depth et Caméra (Depth Anything 3)
- **Estimation de profondeur** : Depth maps haute qualité sur séquences complètes
- **Poses caméra** : Extrinsics/intrinsics pour reconstruction 3D
- **Normales** : Calcul et export des normales de surface
- **Point cloud** : Export PLY global avec couleurs
- **Export Blender** : Génération de script pour export FBX/Alembic

### Exports
- PNG Alpha (séquences)
- PNG RGBA straight ou premultiplied (séquences)
- ProRes4444 (via ffmpeg)
- Depth PNG16 normalisé (séquences)
- Normals PNG (séquences)
- Camera NPZ (intrinsics + extrinsics)
- Point cloud PLY
- Script Blender pour FBX/Alembic

---

## Installation

### Prérequis
- **Python 3.12+** (SAM3 nécessite Python 3.12 minimum)
- **CUDA 12.6+** (recommandé pour performance)
- **PyTorch 2.7+**
- **Git**

### 1. Créer un environnement virtuel

```bash
cd sam3_da3_roto_ultimate_v0_4
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows
```

### 2. Installer les dépendances de base

```bash
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

### 3. Installer SAM3 et Depth Anything 3

**Option A : Script automatique (recommandé)**
```bash
bash install_models.sh
```

**Option B : Installation manuelle**

#### SAM3
```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..
```

#### Depth Anything 3
```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
pip install xformers
pip install -e .
cd ..
```

### 4. Téléchargement des checkpoints

Les modèles seront téléchargés automatiquement depuis HuggingFace lors du premier usage.

**Pour SAM3** :
- `facebook/sam3-hiera-large` (recommandé)
- `facebook/sam3-hiera-base`
- Authentification HuggingFace peut être nécessaire : `huggingface-cli login`

**Pour Depth Anything 3** :
- `depth-anything/DA3-BASE` (léger, rapide)
- `depth-anything/DA3-LARGE` (équilibré)
- `depth-anything/DA3NESTED-GIANT-LARGE` (meilleure qualité, 1.4B params)
- `depth-anything/DA3METRIC-LARGE` (depth métrique)

---

## Utilisation

### Lancer l'application

```bash
python run.py
```

### Workflow typique

#### 1. Import média
- **Vidéo** : Cliquer "📼 Import vidéo" → Sélectionner un fichier .mov/.mp4/.mkv
- **Suite d'images** : Cliquer "🖼️ Import suite" → Sélectionner un dossier contenant des images

#### 2. Charger SAM3
- Entrer le model ID (ex: `facebook/sam3-hiera-large`) ou laisser par défaut
- Cliquer "⚙️ Charger SAM3"
- ⚠️ Le premier chargement peut prendre du temps (téléchargement du checkpoint ~2-3GB)

#### 3. Segmentation

**Mode PCS (Concept) - Image unique :**
1. Sélectionner "Concept (PCS) image" dans le menu déroulant
2. Entrer un prompt texte (ex: "person", "red car", "building")
3. Cliquer "▶ Segment frame"
4. L'application créera automatiquement des objets pour chaque instance détectée

**Mode PVS (Interactif) - Image unique :**
1. Sélectionner "Interactive (PVS) image"
2. Choisir l'outil Point ou Box
3. Choisir le signe + (foreground) ou - (background)
4. Cliquer sur l'image pour ajouter des prompts
5. Cliquer "▶ Segment frame"

**Mode PCS - Tracking vidéo :**
1. Sélectionner "Concept (PCS) video (track all instances)"
2. Entrer un prompt texte
3. Cliquer "🧷 Track (video)"
4. SAM3 propagera automatiquement sur toute la vidéo

**Mode PVS - Tracking vidéo avec keyframes :**
1. Sélectionner "Interactive (PVS) video (keyframes)"
2. Naviguer à une frame et ajouter des points/boxes pour un objet
3. Optionnel : Naviguer à d'autres frames clés et raffiner les prompts
4. Cliquer "🧷 Track (video)"
5. SAM3 interpolera entre les keyframes

#### 4. Refinement Alpha (Onglet Matte)
- Ajuster les sliders pour nettoyer les masques :
  - **Grow/Shrink** : Dilater ou éroder le masque
  - **Fill holes** : Remplir les trous (max area en pixels²)
  - **Remove dots** : Supprimer les petits îlots (max area)
  - **Border fix** : Fermer les bords avec morphologie
  - **Feather** : Adoucir les bords (simple gaussian)
  - **Trimap band** : Raffinage alpha avec distance transform (recommandé pour cheveux)
  - **Temporal smooth** : Stabilisation temporelle (0-100%)

Pour les cheveux/détails fins :
- Activer "Raffiner alpha (trimap distance)"
- Trimap band : 10-25 px
- Feather : 2-6 px
- Temporal : 40-70%

#### 5. RGB Cleanup (Onglet RGB / Comp)
- **Despill** : Supprimer les reflets verts/bleus
  - Green/Blue average : Méthodes simples
  - Physical (auto BG) : Estimation automatique du BG color
- **Edge extend / Pixel spread** : Étendre les pixels RGB aux bords pour éviter les halos noirs
- **Premultiply** : Cocher pour export premultiplied (sinon straight alpha)

#### 6. Depth / Camera (Onglet Depth / Camera DA3)
1. Entrer le model ID DA3 (ex: `depth-anything/DA3-LARGE`)
2. Cliquer "⚙️ Charger DA3"
3. Cliquer "🌊 Depth+Camera (all frames)" pour analyser toute la séquence
4. Prévisualiser :
   - "👁️ Preview depth (false color)" : Visualiser la depth map
   - "👁️ Preview normals" : Visualiser les normales
5. Exporter :
   - Depth PNG16 (séquences 16-bit normalisées)
   - Normals PNG (séquences RGB)
   - Camera NPZ (intrinsics + extrinsics)
   - Point cloud PLY global
   - Script Blender pour export FBX/Alembic

#### 7. Export (Onglet Export)
- **Export alpha PNG** : Séquence alpha pour l'objet actif
- **Export RGBA PNG** : Séquence RGBA avec cleanup RGB appliqué
- **Export alpha ALL objs** : Tous les objets en dossiers séparés
- **Export RGBA ALL objs** : Tous les objets en RGBA
- **Export ProRes4444 MOV** : Vidéo ProRes avec alpha (nécessite ffmpeg)

---

## Raccourcis clavier

- `[` : Frame précédente
- `]` : Frame suivante
- `Ctrl+Enter` : Segment frame
- `Ctrl+T` : Track video

---

## Architecture projet

```
sam3_da3_roto_ultimate_v0_4/
├── run.py                           # Point d'entrée
├── requirements.txt                 # Dépendances Python
├── install_models.sh                # Script d'installation SAM3+DA3
├── README.md                        # Ce fichier
├── sam3roto/
│   ├── app.py                       # Application principale PySide6
│   ├── backend/
│   │   └── sam3_backend.py          # Wrapper SAM3 (PCS/PVS image/vidéo)
│   ├── depth/
│   │   ├── da3_backend.py           # Wrapper Depth Anything 3
│   │   ├── geometry.py              # Utils depth→normals, point cloud
│   │   └── blender_export.py        # Génération script Blender
│   ├── post/
│   │   ├── matte.py                 # Refinement alpha (holes, dots, grow, feather, trimap)
│   │   ├── despill.py               # Despill RGB (green, blue, physical)
│   │   ├── pixelspread.py           # Edge extend / pixel spread
│   │   ├── composite.py             # Premultiply
│   │   └── flowblur.py              # Edge motion blur (optical flow)
│   ├── io/
│   │   ├── media.py                 # Chargement vidéo/séquence
│   │   ├── cache.py                 # Cache masques et depth
│   │   ├── project.py               # Save/load projet
│   │   └── export.py                # Exports PNG/ProRes
│   └── ui/
│       ├── viewer.py                # Viewer interactif avec overlays
│       └── widgets.py               # Widgets custom (LabeledSlider)
```

---

## Troubleshooting

### Erreur "SAM3 n'est pas installé"
- Vérifier que le repo SAM3 est cloné et installé : `pip list | grep sam3`
- Réinstaller : `cd sam3 && pip install -e .`

### Erreur "Depth Anything 3 n'est pas installé"
- Vérifier l'installation : `pip list | grep depth-anything`
- Réinstaller : `cd Depth-Anything-3 && pip install -e .`

### Erreur "ImportError: cannot import name 'Sam3Model' from 'transformers'"
- ⚠️ Ne PAS utiliser l'ancien code qui importe depuis `transformers`
- Le code a été corrigé pour utiliser les imports officiels du repo SAM3
- Importer depuis `sam3.model_builder` et `sam3.model.sam3_image_processor`

### Chargement lent du modèle SAM3
- Le premier chargement télécharge ~2-3GB depuis HuggingFace
- Authentification requise : `huggingface-cli login`
- Les chargements suivants utilisent le cache local

### Out of memory GPU
- Utiliser des modèles plus petits :
  - SAM3 : `facebook/sam3-hiera-base` au lieu de `large`
  - DA3 : `depth-anything/DA3-BASE` au lieu de `GIANT-LARGE`
- Réduire la résolution de la vidéo source
- Pour DA3 point cloud : Augmenter le `stride` dans `_job_da3_export_ply` (ligne 796 app.py)

### Tracking vidéo lent
- Normal pour SAM3 vidéo : ~1-5 fps selon GPU
- Utiliser des séquences plus courtes pour tests
- Considérer découper la vidéo en chunks

### Exports ProRes échouent
- Vérifier que ffmpeg est installé : `ffmpeg -version`
- Installer : `sudo apt install ffmpeg` (Linux) ou via Homebrew (Mac)

---

## Notes de performance (RTX 4090)

- **SAM3 image PCS** : ~0.5-1.0 sec/frame (1920x1080)
- **SAM3 image PVS** : ~0.3-0.7 sec/frame
- **SAM3 vidéo PCS** : ~1-3 fps (dépend de la complexité)
- **DA3-LARGE** : ~0.2-0.5 sec/frame
- **DA3NESTED-GIANT-LARGE** : ~0.8-1.5 sec/frame

Utiliser `bfloat16` (automatique sur GPU CUDA avec support bf16) pour accélérer.

---

## Crédits et Références

### SAM3 (Segment Anything Model 3)
- **Auteur** : Meta AI Research
- **Publication** : Novembre 2025
- **Paper** : https://ai.meta.com/blog/segment-anything-model-3/
- **Code** : https://github.com/facebookresearch/sam3
- **Fonctionnalités** : Promptable Concept Segmentation (PCS), détection et tracking avec prompts texte et visuels

### Depth Anything V3
- **Auteur** : ByteDance Seed Team
- **Publication** : Novembre 2025
- **Paper** : https://arxiv.org/abs/2511.10647
- **Code** : https://github.com/ByteDance-Seed/Depth-Anything-3
- **Site** : https://depth-anything-3.github.io/
- **Fonctionnalités** : Estimation de depth monoculaire, poses caméra, géométrie multi-vue

---

## Licence

Ce projet est un wrapper d'application pour SAM3 et Depth Anything 3.
Consultez les licences des projets originaux :
- SAM3 : Apache 2.0 (Meta AI)
- Depth Anything 3 : Apache 2.0 (ByteDance)

---

## Support

Pour des questions ou bugs :
1. Vérifier les sections Troubleshooting et Installation
2. Consulter les repos officiels SAM3 et DA3
3. Vérifier les logs d'erreur dans le terminal

**Bonne rotoscopie ! 🎬✨**
