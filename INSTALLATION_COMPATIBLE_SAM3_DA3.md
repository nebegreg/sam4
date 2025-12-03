# 🚀 Guide d'Installation Compatible SAM3 + Depth Anything 3 (2025)

**Date**: 2025-12-03
**Objectif**: Installation optimale pour SAM3 Roto Ultimate avec SAM3 et Depth Anything 3

---

## 📋 Table des Matières

1. [Configuration Requise](#configuration-requise)
2. [Versions Compatibles](#versions-compatibles)
3. [Installation Étape par Étape](#installation-étape-par-étape)
4. [Script d'Installation Automatique](#script-dinstallation-automatique)
5. [Vérification](#vérification)
6. [Résolution de Problèmes](#résolution-de-problèmes)

---

## 📊 Configuration Requise

### Matériel

| Composant | Minimum | Recommandé |
|-----------|---------|------------|
| **GPU** | NVIDIA avec CUDA 12.6+ | RTX 3090/4090, A100 |
| **VRAM GPU** | 8GB | 16GB+ (SAM3 848M params) |
| **RAM** | 16GB | 32GB+ |
| **Stockage** | 50GB libre | 100GB+ |
| **CUDA** | 12.6 | 12.8 |

### Système

- **OS**: Linux (Ubuntu 20.04+, Debian 11+) ou Windows 10/11 avec WSL2
- **Architecture**: x86_64 ou ARM64
- **Pilote NVIDIA**: 535+ (pour CUDA 12.6+)

---

## ✅ Versions Compatibles (Validées 2025)

### Configuration Optimale

```yaml
Python:        3.12
PyTorch:       2.7.1
CUDA:          12.8 (ou 12.6)
torchvision:   0.22.1
torchaudio:    2.7.1
transformers:  main branch (git install)
xformers:      latest
numpy:         1.26+
pillow:        10.0+
opencv-python: 4.8+
PySide6:       6.5+
```

### Matrice de Compatibilité

| PyTorch | CUDA | Python | Transformers | SAM3 | DA3 |
|---------|------|--------|--------------|------|-----|
| 2.7.1   | 12.8 | 3.12   | main branch  | ✅   | ✅  |
| 2.7.1   | 12.6 | 3.12   | main branch  | ✅   | ✅  |
| 2.7.0   | 11.8 | 3.12   | main branch  | ✅   | ✅  |
| 2.6.x   | 12.1 | 3.11   | 4.57+        | ⚠️   | ✅  |
| 2.5.x   | 12.1 | 3.11   | 4.55+        | ❌   | ✅  |

**Légende**: ✅ Pleinement compatible | ⚠️ Compatible avec limitations | ❌ Non compatible

---

## 🔧 Installation Étape par Étape

### Étape 1: Vérifier CUDA sur votre Système

```bash
# Vérifier la version CUDA disponible
nvidia-smi

# Vérifier le pilote NVIDIA
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

**Résultat attendu**:
- Driver version: 535+
- CUDA Version: 12.6 ou supérieur

### Étape 2: Créer l'Environnement Virtuel

```bash
# Option A: Avec conda (RECOMMANDÉ)
conda create -n sam3_env python=3.12 -y
conda activate sam3_env

# Option B: Avec venv
python3.12 -m venv ~/venv_sam3_ultimate
source ~/venv_sam3_ultimate/bin/activate
```

**⚠️ IMPORTANT**: Vérifiez que l'environnement est activé:
```bash
which python3
# DOIT afficher: /home/votre_user/anaconda3/envs/sam3_env/bin/python3
# OU: /home/votre_user/venv_sam3_ultimate/bin/python3
# PAS: /usr/bin/python3
```

### Étape 3: Installer PyTorch 2.7.1 avec CUDA

**Option A: CUDA 12.8 (Recommandé - Support Blackwell)**
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
```

**Option B: CUDA 12.6**
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126
```

**Option C: CUDA 11.8 (Si hardware plus ancien)**
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

**Vérification PyTorch**:
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

**Résultat attendu**:
```
PyTorch: 2.7.1
CUDA available: True
CUDA version: 12.8 (ou 12.6, ou 11.8)
```

### Étape 4: Installer les Dépendances de Base

```bash
# Dépendances essentielles
pip install numpy pillow opencv-python matplotlib scipy

# Interface graphique Qt
pip install PySide6

# Outils ML/CV
pip install scikit-image scikit-learn einops timm
```

### Étape 5: Installer xformers (Requis pour Depth Anything 3)

```bash
pip install xformers
```

**Vérification**:
```bash
python3 -c "import xformers; print(f'xformers: {xformers.__version__}')"
```

### Étape 6: Installer Transformers (avec support SAM3)

**⚠️ IMPORTANT**: SAM3 nécessite la version main branch (pas encore dans version stable)

```bash
# Installer depuis GitHub (main branch)
pip install git+https://github.com/huggingface/transformers.git

# Installer les dépendances de transformers
pip install accelerate sentencepiece protobuf
```

**Vérification**:
```bash
python3 -c "from transformers import Sam3Model, Sam3Processor; print('SAM3 support: OK')"
```

Si erreur `ImportError: cannot import name 'Sam3Model'`:
- Attendez la prochaine release de transformers OU
- Réinstallez: `pip install --force-reinstall git+https://github.com/huggingface/transformers.git`

### Étape 7: Installer SAM3 (Facebook Research)

```bash
# Cloner le repo SAM3
cd /tmp
git clone https://github.com/facebookresearch/sam3.git
cd sam3

# Installer SAM3
pip install -e .

# Optionnel: Installer avec notebooks
pip install -e ".[notebooks]"
```

**⚠️ Authentification Requise**:
```bash
# Installer huggingface_hub
pip install huggingface-hub

# Se connecter (nécessite un token HuggingFace)
huggingface-cli login
```

**Obtenir un token**:
1. Aller sur https://huggingface.co/settings/tokens
2. Créer un nouveau token (read access)
3. Demander l'accès au repo: https://huggingface.co/facebook/sam3
4. Utiliser le token pour se connecter

**Vérification**:
```bash
python3 -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 import: OK')"
```

### Étape 8: Installer Depth Anything 3

```bash
# Cloner le repo Depth Anything 3
cd /tmp
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3

# Installation de base
pip install -e .

# OU installation complète avec toutes les fonctionnalités
pip install -e ".[all]"

# Optionnel: Support Gaussian 3D
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

**Vérification**:
```bash
python3 -c "from depth_anything_3.api import DepthAnything3; print('Depth Anything 3: OK')"
```

### Étape 9: Installer les Dépendances de l'Application

```bash
# Aller dans le dossier du projet
cd ~/Downloads/sam4-main  # Ou votre chemin

# Installer les dépendances additionnelles
pip install pytest pytest-cov

# Vérifier requirements.txt s'il existe
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi
```

### Étape 10: Vérification Finale

```bash
# Lancer le script de diagnostic
python3 diagnostic.py
```

**Résultat attendu**: Tous les tests doivent passer ✅

---

## 🤖 Script d'Installation Automatique

Créez un fichier `install_sam3_da3.sh`:

```bash
#!/bin/bash

# Installation automatique pour SAM3 + Depth Anything 3
# Date: 2025-12-03
# Compatible: Python 3.12, PyTorch 2.7.1, CUDA 12.8/12.6

set -e  # Arrêter en cas d'erreur

echo "=== Installation SAM3 + Depth Anything 3 ==="
echo ""

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction de vérification
check_command() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
    else
        echo -e "${RED}✗ $1 - ÉCHEC${NC}"
        exit 1
    fi
}

# 1. Vérifier CUDA
echo -e "${YELLOW}[1/11] Vérification CUDA...${NC}"
nvidia-smi > /dev/null 2>&1
check_command "CUDA disponible"

# 2. Vérifier Python
echo -e "${YELLOW}[2/11] Vérification Python 3.12...${NC}"
python3 --version | grep "3.12"
check_command "Python 3.12"

# 3. Créer l'environnement
echo -e "${YELLOW}[3/11] Création environnement virtuel...${NC}"
if command -v conda &> /dev/null; then
    conda create -n sam3_env python=3.12 -y
    eval "$(conda shell.bash hook)"
    conda activate sam3_env
else
    python3.12 -m venv ~/venv_sam3_ultimate
    source ~/venv_sam3_ultimate/bin/activate
fi
check_command "Environnement créé"

# 4. Installer PyTorch 2.7.1 avec CUDA 12.8
echo -e "${YELLOW}[4/11] Installation PyTorch 2.7.1 + CUDA 12.8...${NC}"
pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
check_command "PyTorch installé"

# Vérifier PyTorch CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
check_command "PyTorch CUDA opérationnel"

# 5. Installer dépendances de base
echo -e "${YELLOW}[5/11] Installation dépendances de base...${NC}"
pip install numpy pillow opencv-python matplotlib scipy \
    PySide6 scikit-image scikit-learn einops timm
check_command "Dépendances de base installées"

# 6. Installer xformers
echo -e "${YELLOW}[6/11] Installation xformers...${NC}"
pip install xformers
check_command "xformers installé"

# 7. Installer Transformers (main branch)
echo -e "${YELLOW}[7/11] Installation Transformers (main)...${NC}"
pip install git+https://github.com/huggingface/transformers.git
pip install accelerate sentencepiece protobuf huggingface-hub
check_command "Transformers installé"

# 8. Installer SAM3
echo -e "${YELLOW}[8/11] Installation SAM3...${NC}"
cd /tmp
if [ -d "sam3" ]; then rm -rf sam3; fi
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
check_command "SAM3 installé"

# 9. Installer Depth Anything 3
echo -e "${YELLOW}[9/11] Installation Depth Anything 3...${NC}"
cd /tmp
if [ -d "Depth-Anything-3" ]; then rm -rf Depth-Anything-3; fi
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
pip install -e ".[all]"
check_command "Depth Anything 3 installé"

# 10. Installer pytest
echo -e "${YELLOW}[10/11] Installation outils de test...${NC}"
pip install pytest pytest-cov
check_command "Pytest installé"

# 11. Vérification finale
echo -e "${YELLOW}[11/11] Vérification finale...${NC}"
echo ""
echo "=== Versions Installées ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python3 -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"
python3 -c "import xformers; print(f'xformers: {xformers.__version__}')"
python3 -c "import numpy; print(f'numpy: {numpy.__version__}')"
python3 -c "import PySide6; print(f'PySide6: OK')"
python3 -c "import cv2; print(f'opencv: {cv2.__version__}')"
echo ""

# Test imports
echo "=== Test des Imports ==="
python3 -c "from transformers import Sam3Model, Sam3Processor; print('✓ SAM3 (transformers)')" || echo "⚠ SAM3 (transformers) - Utiliser sam3 GitHub à la place"
python3 -c "from sam3.model_builder import build_sam3_image_model; print('✓ SAM3 (GitHub repo)')"
python3 -c "from depth_anything_3.api import DepthAnything3; print('✓ Depth Anything 3')"
echo ""

echo -e "${GREEN}=== ✓ Installation Terminée avec Succès! ===${NC}"
echo ""
echo "Pour activer l'environnement:"
if command -v conda &> /dev/null; then
    echo "  conda activate sam3_env"
else
    echo "  source ~/venv_sam3_ultimate/bin/activate"
fi
echo ""
echo "IMPORTANT: Authentification HuggingFace requise pour SAM3:"
echo "  huggingface-cli login"
echo "  Token: https://huggingface.co/settings/tokens"
echo "  Demander accès: https://huggingface.co/facebook/sam3"
echo ""
```

**Utilisation**:
```bash
chmod +x install_sam3_da3.sh
./install_sam3_da3.sh
```

---

## 🧪 Vérification de l'Installation

### Script de Test Complet

Créez `test_installation.py`:

```python
#!/usr/bin/env python3
"""
Test complet de l'installation SAM3 + Depth Anything 3
"""

import sys

def test_imports():
    """Test tous les imports requis"""
    print("=== Test des Imports ===\n")

    tests = [
        ("PyTorch", "import torch; print(f'  Version: {torch.__version__}'); assert torch.cuda.is_available(), 'CUDA non disponible'"),
        ("torchvision", "import torchvision; print(f'  Version: {torchvision.__version__}')"),
        ("NumPy", "import numpy; print(f'  Version: {numpy.__version__}')"),
        ("Pillow", "import PIL; print(f'  Version: {PIL.__version__}')"),
        ("OpenCV", "import cv2; print(f'  Version: {cv2.__version__}')"),
        ("PySide6", "import PySide6; print('  Version: OK')"),
        ("xformers", "import xformers; print(f'  Version: {xformers.__version__}')"),
        ("transformers", "import transformers; print(f'  Version: {transformers.__version__}')"),
        ("einops", "import einops; print('  Version: OK')"),
        ("timm", "import timm; print(f'  Version: {timm.__version__}')"),
    ]

    passed = 0
    failed = 0

    for name, code in tests:
        try:
            print(f"[TEST] {name}...", end=" ")
            exec(code)
            print("✓")
            passed += 1
        except Exception as e:
            print(f"✗ ÉCHEC: {e}")
            failed += 1

    print(f"\n✓ Réussis: {passed}/{len(tests)}")
    if failed > 0:
        print(f"✗ Échecs: {failed}/{len(tests)}")

    return failed == 0

def test_sam3():
    """Test SAM3"""
    print("\n=== Test SAM3 ===\n")

    try:
        print("[TEST] Import transformers SAM3...", end=" ")
        from transformers import Sam3Model, Sam3Processor
        print("✓")
        print("  Méthode: transformers")
    except ImportError:
        print("✗ (Non disponible)")
        print("  Note: Utiliser sam3 GitHub repo")

    try:
        print("[TEST] Import sam3 GitHub...", end=" ")
        from sam3.model_builder import build_sam3_image_model
        print("✓")
        print("  Méthode: GitHub repo")
        return True
    except ImportError as e:
        print(f"✗ ÉCHEC: {e}")
        return False

def test_depth_anything():
    """Test Depth Anything 3"""
    print("\n=== Test Depth Anything 3 ===\n")

    try:
        print("[TEST] Import Depth Anything 3...", end=" ")
        from depth_anything_3.api import DepthAnything3
        print("✓")
        return True
    except ImportError as e:
        print(f"✗ ÉCHEC: {e}")
        return False

def test_cuda():
    """Test CUDA"""
    print("\n=== Test CUDA ===\n")

    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  VRAM totale: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")

        return True
    else:
        print("⚠ ATTENTION: CUDA n'est pas disponible!")
        return False

def main():
    print("=" * 60)
    print("Test d'Installation SAM3 + Depth Anything 3")
    print("=" * 60)

    results = []

    results.append(("Imports de base", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("SAM3", test_sam3()))
    results.append(("Depth Anything 3", test_depth_anything()))

    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)

    for name, passed in results:
        status = "✓ OK" if passed else "✗ ÉCHEC"
        print(f"{name:.<40} {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n🎉 Installation complète et fonctionnelle!")
        return 0
    else:
        print("\n⚠ Certains composants ont échoué")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

**Lancer le test**:
```bash
python3 test_installation.py
```

---

## 🔍 Résolution de Problèmes

### Problème 1: `ImportError: cannot import name 'Sam3Model'`

**Cause**: Version de transformers trop ancienne

**Solution**:
```bash
pip uninstall transformers -y
pip install --force-reinstall git+https://github.com/huggingface/transformers.git
```

### Problème 2: `CUDA not available` malgré nvidia-smi OK

**Cause**: PyTorch installé sans support CUDA

**Solution**:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
```

**Vérifier**:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Problème 3: `ModuleNotFoundError: No module named 'sam3'`

**Cause**: SAM3 GitHub repo pas installé

**Solution**:
```bash
cd /tmp
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### Problème 4: Erreur lors du téléchargement des modèles SAM3

**Cause**: Pas authentifié sur HuggingFace

**Solution**:
```bash
# 1. Créer un token sur https://huggingface.co/settings/tokens
# 2. Demander l'accès: https://huggingface.co/facebook/sam3
# 3. Se connecter
huggingface-cli login
# Coller le token
```

### Problème 5: `xformers` ne s'installe pas

**Cause**: Incompatibilité PyTorch/CUDA

**Solution**:
```bash
# Vérifier la version PyTorch
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"

# Réinstaller xformers
pip uninstall xformers -y
pip install xformers --no-deps
pip install xformers
```

### Problème 6: Segmentation fault au lancement

**Cause**: Problème Qt threading (déjà corrigé dans app.py)

**Vérification**:
```bash
# Vérifier que le fix est appliqué
grep "_active_threads" sam3roto/app.py
```

Devrait afficher:
```python
self._active_threads: List[Tuple[QtCore.QThread, Worker]] = []
```

### Problème 7: Mémoire GPU insuffisante

**Symptôme**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Utiliser un modèle plus petit (Base au lieu de Large)
2. Réduire la taille des images
3. Activer le mixed precision (FP16)
4. Vider le cache CUDA:
```python
import torch
torch.cuda.empty_cache()
```

### Problème 8: Environnement virtuel pas activé

**Symptôme**: `(sam3)` visible mais `which python3` = `/usr/bin/python3`

**Solution**:
```bash
# Désactiver l'ancien
deactivate  # ou conda deactivate

# Réactiver correctement
source ~/venv_sam3_ultimate/bin/activate
# OU
conda activate sam3_env

# Vérifier
which python3  # DOIT pointer vers le venv!
```

---

## 📚 Sources et Documentation

### SAM3 (Segment Anything Model 3)
- [GitHub - facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- [SAM3 on Hugging Face](https://huggingface.co/facebook/sam3)
- [SAM3 Documentation (Hugging Face)](https://huggingface.co/docs/transformers/main/model_doc/sam3)
- [Meta AI - SAM3 Announcement](https://ai.meta.com/blog/segment-anything-model-3/)

### Depth Anything 3
- [GitHub - ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [Depth Anything 3 Guide 2025](https://apatero.com/blog/depth-anything-v3-complete-guide-use-cases-2025)
- [Depth Anything 3 Project Page](https://depth-anything-3.github.io/)

### PyTorch & CUDA
- [PyTorch 2.7 Release Notes](https://pytorch.org/blog/pytorch-2-7/)
- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)
- [PyTorch CUDA Compatibility Matrix](https://github.com/eminsafa/pytorch-cuda-compatibility)

### Transformers
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Transformers Releases](https://github.com/huggingface/transformers/releases)

---

## 🎯 Checklist Finale

Avant de lancer l'application, vérifiez:

- [ ] GPU NVIDIA avec CUDA 12.6+ détecté (`nvidia-smi`)
- [ ] Python 3.12 installé
- [ ] Environnement virtuel créé et **ACTIVÉ**
- [ ] `which python3` pointe vers le venv (PAS `/usr/bin/python3`)
- [ ] PyTorch 2.7.1 installé avec support CUDA
- [ ] `torch.cuda.is_available()` retourne `True`
- [ ] xformers installé
- [ ] Transformers (main branch) installé
- [ ] SAM3 (GitHub repo) installé
- [ ] Depth Anything 3 installé
- [ ] PySide6 installé
- [ ] Authentification HuggingFace effectuée
- [ ] Accès au repo SAM3 accordé
- [ ] Test `python3 test_installation.py` réussit

**Si toutes les cases sont cochées → Vous pouvez lancer l'application!**

```bash
cd ~/Downloads/sam4-main
python3 run.py
```

---

## 🚀 Prochaines Étapes

1. **Lancer le script d'installation**:
   ```bash
   ./install_sam3_da3.sh
   ```

2. **S'authentifier sur HuggingFace**:
   ```bash
   huggingface-cli login
   ```

3. **Tester l'installation**:
   ```bash
   python3 test_installation.py
   ```

4. **Lancer l'application**:
   ```bash
   python3 run.py
   ```

---

**Dernière mise à jour**: 2025-12-03
**Validé avec**: SAM3 (Nov 2025), Depth Anything 3 (Nov 2025), PyTorch 2.7.1 (Apr 2025)
