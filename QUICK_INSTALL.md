# Installation Rapide - SAM3 + Depth Anything 3

## ⚡ Installation en 1 commande

```bash
cd /home/user/sam4
./setup_venv_fixed.sh
```

## 📋 Prérequis

- ✅ CUDA et drivers NVIDIA installés
- ✅ Python 3.12 disponible
- ✅ 20GB d'espace disque libre
- ✅ Connexion internet stable

## 🔍 Vérifications avant installation

### 1. Vérifier CUDA
```bash
nvidia-smi
# Doit afficher les informations de votre GPU
```

### 2. Vérifier Python 3.12
```bash
python3.12 --version
# Doit afficher: Python 3.12.x
```

### 3. Vérifier l'espace disque
```bash
df -h ~
# Minimum 20GB libres recommandés
```

## 🚀 Installation

### Option 1: Installation automatique (recommandée)
```bash
cd /home/user/sam4
./setup_venv_fixed.sh
```

Le script va:
1. ✅ Vérifier CUDA et Python 3.12
2. ✅ Créer un environnement virtuel propre
3. ✅ Installer PyTorch 2.7.1 avec CUDA
4. ✅ Installer toutes les dépendances dans le bon ordre
5. ✅ Installer SAM3 et Depth Anything 3
6. ✅ Tester l'installation

**Durée estimée:** 15-30 minutes

### Option 2: Installation manuelle

Si vous préférez contrôler chaque étape:

```bash
# 1. Créer le venv
python3.12 -m venv ~/venv_sam3_fixed

# 2. Activer le venv
source ~/venv_sam3_fixed/bin/activate

# 3. Mettre à jour pip
pip install --upgrade pip setuptools wheel

# 4. Installer PyTorch (OBLIGATOIRE EN PREMIER)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 5. Installer les dépendances de base
pip install numpy pillow opencv-python scipy matplotlib \
    scikit-image scikit-learn PySide6 einops timm

# 6. Installer xformers
pip install xformers --no-build-isolation

# 7. Installer Hugging Face
pip install huggingface-hub accelerate sentencepiece protobuf
pip install git+https://github.com/huggingface/transformers.git

# 8. Installer SAM3
cd /tmp
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# 9. Installer Depth Anything 3
cd /tmp
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
pip install -e . --no-deps
pip install evo pycocotools decord pre-commit

# 10. Retour au projet
cd /home/user/sam4
```

## ⚠️ Problèmes courants

### Erreur: "ModuleNotFoundError: No module named 'torch'"
**Solution:** PyTorch doit être installé AVANT xformers
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install xformers --no-build-isolation
```

### Erreur: "CUDA not available"
**Solution:** Vérifier les drivers NVIDIA
```bash
nvidia-smi
# Si ça ne marche pas, réinstaller les drivers NVIDIA
```

### Erreur: "pip is still looking at multiple versions of xformers"
**Solution:** Utiliser le script corrigé qui installe dans le bon ordre

### Erreur: "No space left on device"
**Solution:** Libérer de l'espace disque
```bash
# Supprimer les anciens venvs
rm -rf ~/venv_old

# Nettoyer le cache pip
pip cache purge
```

## ✅ Après l'installation

### 1. Activer l'environnement
```bash
source ~/venv_sam3_fixed/bin/activate
# Ou utilisez le script rapide:
./activate_venv.sh
```

### 2. Configurer HuggingFace (OBLIGATOIRE)
```bash
# Obtenir un token sur https://huggingface.co/settings/tokens
huggingface-cli login

# Demander l'accès à SAM3 sur https://huggingface.co/facebook/sam3
```

### 3. Tester l'installation
```bash
python3 test_installation.py
```

### 4. Lancer l'application
```bash
python3 run.py
```

## 📊 Vérifier l'installation

```bash
source ~/venv_sam3_fixed/bin/activate

# Vérifier PyTorch et CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Vérifier SAM3
python3 -c "from sam3.model_builder import build_sam3_image_model; print('✓ SAM3')"

# Vérifier Depth Anything 3
python3 -c "import depth_anything_3; print('✓ Depth Anything 3')"
```

## 🔄 Réinstallation

Si l'installation échoue, supprimez l'ancien environnement et recommencez:

```bash
# Supprimer l'ancien venv
rm -rf ~/venv_sam3_fixed

# Réinstaller
./setup_venv_fixed.sh
```

## 📝 Notes importantes

### Ordre d'installation CRITIQUE

L'ordre d'installation des packages est **CRUCIAL** pour éviter les erreurs:

1. **pip, setuptools, wheel** (outils de base)
2. **PyTorch** (DOIT être installé en premier ⚠️)
3. **Bibliothèques de base** (numpy, opencv, etc.)
4. **xformers** (nécessite PyTorch ⚠️)
5. **Transformers** (nécessite PyTorch)
6. **SAM3** (nécessite Transformers)
7. **Depth Anything 3** (nécessite tout le reste)

❌ **NE PAS installer dans le désordre**
✅ **Suivre cet ordre strictement**

## 🆘 Besoin d'aide?

### Logs détaillés
```bash
./setup_venv_fixed.sh 2>&1 | tee installation.log
```

### Diagnostics
```bash
# Vérifier Python
which python3
python3 --version

# Vérifier CUDA
nvidia-smi

# Vérifier l'espace disque
df -h

# Vérifier les packages installés
pip list | grep -E "torch|sam3|depth"
```

### Documentation complète
- `VENV_FIX_EXPLANATION.md` - Explications détaillées des corrections
- `INSTALLATION_ROCKY_LINUX.md` - Installation sur Rocky Linux
- `README.md` - Documentation générale

## 🎯 Checklist de vérification

- [ ] CUDA fonctionne (`nvidia-smi`)
- [ ] Python 3.12 installé (`python3.12 --version`)
- [ ] 20GB d'espace disque libre
- [ ] Script d'installation exécuté
- [ ] Environnement activé
- [ ] HuggingFace configuré
- [ ] Tests passent avec succès
- [ ] Application se lance

## 🔗 Liens utiles

- [HuggingFace Tokens](https://huggingface.co/settings/tokens)
- [SAM3 Access](https://huggingface.co/facebook/sam3)
- [PyTorch CUDA](https://pytorch.org/get-started/locally/)
- [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx)

---

**Version:** 2025-12-03
**Python:** 3.12
**PyTorch:** 2.7.1
**CUDA:** 12.8 / 12.6
