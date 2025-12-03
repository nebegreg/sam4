# 🚀 Quick Start - Installation SAM3 + Depth Anything 3

Installation rapide en 3 étapes pour SAM3 Roto Ultimate.

---

## ⚡ Installation Rapide (Recommandée)

### Étape 1: Lancer le script d'installation

```bash
cd ~/Downloads/sam4-main  # Ou votre dossier
chmod +x install_sam3_da3.sh
./install_sam3_da3.sh
```

Ce script installe automatiquement:
- ✅ Python 3.12 + environnement virtuel
- ✅ PyTorch 2.7.1 + CUDA 12.8
- ✅ Transformers (main branch avec SAM3)
- ✅ SAM3 (Facebook Research)
- ✅ Depth Anything 3 (ByteDance)
- ✅ Toutes les dépendances

**Durée**: 10-15 minutes (selon connexion internet)

### Étape 2: Authentification HuggingFace

```bash
# 1. Créer un token: https://huggingface.co/settings/tokens
# 2. Demander l'accès: https://huggingface.co/facebook/sam3
# 3. Se connecter
huggingface-cli login
# Coller votre token
```

### Étape 3: Tester et Lancer

```bash
# Tester l'installation
python3 test_installation.py

# Si tout est OK ✓, lancer l'application
python3 run.py
```

---

## 📚 Documentation Complète

Pour plus de détails, voir:

- **[INSTALLATION_COMPATIBLE_SAM3_DA3.md](INSTALLATION_COMPATIBLE_SAM3_DA3.md)** - Guide complet (21 KB)
  - Configuration requise détaillée
  - Installation étape par étape
  - Résolution de tous les problèmes courants

- **[CODE_ANALYSIS_REPORT.md](CODE_ANALYSIS_REPORT.md)** - Analyse du code
  - Vérification syntaxe (44 fichiers ✓)
  - Diagnostic des problèmes
  - Solutions détaillées

---

## ⚙️ Configuration Recommandée

| Composant | Version |
|-----------|---------|
| Python | 3.12 |
| PyTorch | 2.7.1 |
| CUDA | 12.8 (ou 12.6) |
| Transformers | main branch |
| GPU VRAM | 16GB+ |
| RAM | 32GB+ |

---

## 🔧 Installation Manuelle

Si vous préférez installer manuellement:

```bash
# 1. Créer environnement
python3.12 -m venv ~/venv_sam3_ultimate
source ~/venv_sam3_ultimate/bin/activate

# 2. Installer PyTorch
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. Installer dépendances
pip install numpy pillow opencv-python PySide6 xformers einops timm

# 4. Installer Transformers (main)
pip install git+https://github.com/huggingface/transformers.git

# 5. Installer SAM3
cd /tmp
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e .

# 6. Installer Depth Anything 3
cd /tmp
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3 && pip install -e ".[all]"

# 7. Authentification
huggingface-cli login
```

---

## ✅ Vérification

Avant de lancer l'application:

```bash
# Vérifier que l'environnement est activé
which python3
# DOIT afficher: /home/votre_user/venv_sam3_ultimate/bin/python3
# PAS: /usr/bin/python3

# Vérifier CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
# DOIT afficher: True

# Test complet
python3 test_installation.py
```

---

## 🆘 Problèmes Courants

### Problème: `CUDA not available`

**Solution**:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
```

### Problème: `ModuleNotFoundError: No module named 'numpy'`

**Solution**: Environnement pas activé
```bash
source ~/venv_sam3_ultimate/bin/activate
which python3  # Vérifier le path
```

### Problème: `ImportError: cannot import name 'Sam3Model'`

**Solution**: Transformers trop ancien
```bash
pip install --force-reinstall git+https://github.com/huggingface/transformers.git
```

### Problème: Segmentation fault

**Solution**: Déjà corrigé dans `sam3roto/app.py`
- Vérifier: `grep "_active_threads" sam3roto/app.py`
- Si absent, voir [SEGFAULT_FIX_GUIDE.md](SEGFAULT_FIX_GUIDE.md)

---

## 📖 Guides Disponibles

| Guide | Description | Taille |
|-------|-------------|--------|
| **QUICK_START.md** | Ce fichier - Installation rapide | 4 KB |
| **INSTALLATION_COMPATIBLE_SAM3_DA3.md** | Guide complet d'installation | 21 KB |
| **CODE_ANALYSIS_REPORT.md** | Analyse complète du code | 13 KB |
| **GUIDE_COMPLET_LANCEMENT.md** | Guide de lancement | 10 KB |
| **SEGFAULT_FIX_GUIDE.md** | Résolution segfaults | 10 KB |

---

## 🎯 Checklist Rapide

Avant de lancer `python3 run.py`:

- [ ] GPU NVIDIA détecté (`nvidia-smi`)
- [ ] Environnement activé (`which python3` → venv)
- [ ] PyTorch avec CUDA (`torch.cuda.is_available()` = True)
- [ ] SAM3 installé (GitHub repo)
- [ ] Depth Anything 3 installé
- [ ] HuggingFace authentifié
- [ ] Test installation OK (`python3 test_installation.py`)

**Si toutes les cases cochées → Lancer l'application!**

---

## 📞 Support

En cas de problème:

1. Lire **INSTALLATION_COMPATIBLE_SAM3_DA3.md** section "Résolution de Problèmes"
2. Lancer `python3 diagnostic.py` pour identifier le problème
3. Vérifier que l'environnement virtuel est activé
4. Réinstaller si nécessaire: `./install_sam3_da3.sh`

---

**Installation validée**: 2025-12-03
**Compatibilité**: SAM3 (Nov 2025) + Depth Anything 3 (Nov 2025) + PyTorch 2.7.1 (Apr 2025)
