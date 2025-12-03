# Correction du Tracking Vidéo SAM3

## ❌ Problème identifié

L'application crashait lors du tracking vidéo avec l'erreur:

```
AttributeError: 'Sam3Model' object has no attribute 'handle_request'
```

### Cause

Le backend SAM3 chargeait le modèle depuis **transformers/HuggingFace** en priorité, mais cette version n'a pas l'API vidéo complète. La méthode `handle_request()` existe uniquement dans l'implémentation **GitHub officielle** de SAM3.

```python
# Ce qui se passait avant:
self._video_predictor = Sam3Model  # De transformers - PAS d'API vidéo!
response = self._video_predictor.handle_request(...)  # ❌ CRASH
```

## ✅ Solution implémentée

### 1. Inversion de l'ordre de chargement

Le backend essaie maintenant en priorité le **repo GitHub officiel** qui a l'API vidéo complète:

```
Méthode 1 (PRIORITAIRE): Repo GitHub → Support vidéo complet ✅
Méthode 2 (FALLBACK): Transformers → Image seulement ⚠️
```

### 2. Avertissements clairs

Si le système tombe en fallback sur transformers, il affiche:

```
⚠️  AVERTISSEMENT: Le tracking vidéo ne fonctionnera PAS avec transformers
⚠️  Pour la vidéo, installez le repo GitHub avec:
     ./install_sam3_github.sh
```

### 3. Script d'installation automatique

Nouveau script: `install_sam3_github.sh` qui:
- Vérifie que le venv est activé
- Vérifie PyTorch et Transformers
- Clone et installe SAM3 depuis GitHub
- Vérifie que l'API vidéo est disponible

## 🚀 Pour corriger votre installation

### Avec le script automatique (RECOMMANDÉ)

```bash
# 1. Activer votre environnement
source ~/venv_sam3_fixed/bin/activate
# OU
conda activate sam3_da3

# 2. Installer SAM3 GitHub
./install_sam3_github.sh

# 3. Lancer l'application
python3 run.py
```

### Installation manuelle

Si vous préférez installer manuellement:

```bash
# 1. Activer l'environnement
conda activate sam3_da3

# 2. Installer SAM3 GitHub
cd /tmp
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# 3. Vérifier l'installation
python3 -c "from sam3.model_builder import build_sam3_video_predictor; print('✓ OK')"

# 4. Retourner au projet et lancer
cd /home/reepost/Downloads/sam4-main
python3 run.py
```

## 📋 Vérification

Après installation, au lancement vous devriez voir:

```
[SAM3] 🔄 Tentative 1: Repo GitHub officiel...
[SAM3] ✓ Imports repo GitHub réussis
[SAM3] ✓ Image model OK
[SAM3] ✓ Video predictor OK
✅ SAM3 chargé avec succès (repo GitHub)
```

Au lieu de:

```
[SAM3] 🔄 Tentative 1: Transformers/HuggingFace...
✅ SAM3 chargé avec succès (transformers)  ← Pas bon pour la vidéo!
```

## 🔍 Détails techniques

### API Transformers vs GitHub

| Fonctionnalité | Transformers | GitHub |
|----------------|--------------|--------|
| Segmentation image (PCS) | ✅ | ✅ |
| Segmentation interactive (PVS) | ✅ | ✅ |
| Tracking vidéo | ❌ | ✅ |
| Méthode `handle_request()` | ❌ | ✅ |
| API session vidéo | ❌ | ✅ |
| Propagation temporelle | ❌ | ✅ |

### Méthodes vidéo requises

Le tracking vidéo utilise ces méthodes (GitHub seulement):

```python
# Start session
response = predictor.handle_request({
    "type": "start_session",
    "resource_path": "/path/to/frames"
})

# Add prompts
response = predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,
    "text": "person"
})

# Propagate
response = predictor.handle_request({
    "type": "propagate",
    "session_id": session_id
})

# Get masks
response = predictor.handle_request({
    "type": "get_masks",
    "session_id": session_id,
    "frame_index": i
})
```

Ces méthodes n'existent PAS dans `Sam3Model` de transformers.

## 📚 Fichiers modifiés

### `sam3roto/backend/sam3_backend.py`

**Changements:**
1. Inversion de l'ordre de chargement (GitHub en premier)
2. Ajout d'avertissements clairs si fallback sur transformers
3. Mise à jour des messages d'erreur

**Avant:**
```python
# MÉTHODE 1: Transformers
# MÉTHODE 2: GitHub
```

**Après:**
```python
# MÉTHODE 1: Repo GitHub (PRIORITAIRE)
# MÉTHODE 2: Transformers (FALLBACK - image seulement)
```

### Nouveaux fichiers

1. **`install_sam3_github.sh`** - Script d'installation automatique
2. **`FIX_VIDEO_TRACKING.md`** - Ce document

## ⚠️ Notes importantes

### Pour les nouveaux utilisateurs

Si vous installez pour la première fois, utilisez:
```bash
./setup_venv_fixed.sh  # Crée l'environnement complet
./install_sam3_github.sh  # Ajoute le support vidéo
```

### Pour les utilisateurs existants

Si vous avez déjà un environnement:
```bash
conda activate sam3_da3  # Ou votre environnement
./install_sam3_github.sh  # Ajoute juste le support vidéo
```

### Compatibilité

- ✅ Compatible avec Python 3.10, 3.11, 3.12
- ✅ Compatible avec PyTorch 2.7.x
- ✅ Compatible avec transformers 4.x et 5.x
- ✅ Les deux implémentations peuvent coexister

## 🐛 Dépannage

### "No module named 'sam3'"

SAM3 GitHub n'est pas installé:
```bash
./install_sam3_github.sh
```

### "No module named 'transformers'"

Transformers n'est pas installé:
```bash
pip install git+https://github.com/huggingface/transformers.git
```

### "pycocotools not found"

Dépendances manquantes:
```bash
pip install pycocotools decord
```

### Le tracking vidéo crash toujours

Vérifiez que le repo GitHub est bien chargé:
```bash
python3 -c "from sam3.model_builder import build_sam3_video_predictor; print('OK')"
```

Si ça échoue, réinstallez:
```bash
cd /tmp
rm -rf sam3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

## 📖 Références

- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
- [SAM3 HuggingFace](https://huggingface.co/facebook/sam3)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

---

**Date:** 2025-12-03
**Version:** 1.0
**Status:** ✅ Résolu
