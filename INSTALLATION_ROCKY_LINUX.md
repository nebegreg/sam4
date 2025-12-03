# 🚀 Installation pour Rocky Linux 9.5

**Date**: 2025-12-03
**Compatible**: Rocky Linux, RHEL, CentOS, AlmaLinux, Fedora

---

## ⚠️ Important pour Rocky Linux

Rocky Linux 9.5 utilise **`dnf`** comme gestionnaire de paquets (pas `apt`).

Trois scripts sont disponibles:

1. **`install_sam3_da3_universal.sh`** ⭐ **RECOMMANDÉ** - Détection automatique
2. **`install_sam3_da3_rocky.sh`** - Spécifique Rocky/RHEL
3. **`install_sam3_da3.sh`** - Pour Debian/Ubuntu uniquement (ne PAS utiliser)

---

## 🎯 Installation Rapide (Recommandée)

### Option 1: Script Universel (Détection Automatique)

```bash
cd ~/Downloads/sam4-main  # Ou votre dossier

# Rendre le script exécutable
chmod +x install_sam3_da3_universal.sh

# Lancer l'installation
./install_sam3_da3_universal.sh
```

Ce script détecte automatiquement Rocky Linux et utilise `dnf`.

### Option 2: Script Spécifique Rocky Linux

```bash
# Rendre le script exécutable
chmod +x install_sam3_da3_rocky.sh

# Lancer l'installation
./install_sam3_da3_rocky.sh
```

---

## 📋 Prérequis Rocky Linux

### 1. Pilotes NVIDIA

```bash
# Vérifier si les pilotes sont installés
nvidia-smi

# Si non installé:
sudo dnf install -y epel-release
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install -y nvidia-driver nvidia-settings cuda
sudo reboot
```

### 2. Outils de Développement

```bash
# Installer les outils de base
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y git wget curl
```

### 3. EPEL et CRB

```bash
# Activer EPEL (Extra Packages for Enterprise Linux)
sudo dnf install -y epel-release

# Activer CRB (CodeReady Builder) pour Rocky 9
sudo dnf config-manager --set-enabled crb
```

---

## 🔧 Installation Manuelle (Si script échoue)

### Étape 1: Installer Python 3.12

#### Option A: Depuis les dépôts (si disponible)

```bash
sudo dnf install -y python3.12 python3.12-devel python3.12-pip
```

#### Option B: Compilation depuis les sources

```bash
# Installer les dépendances
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y openssl-devel bzip2-devel libffi-devel zlib-devel wget

# Télécharger Python 3.12.7
cd /tmp
wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz
tar xzf Python-3.12.7.tgz
cd Python-3.12.7

# Compiler
./configure --enable-optimizations --with-ensurepip=install
make -j $(nproc)
sudo make altinstall

# Vérifier
python3.12 --version
```

### Étape 2: Créer Environnement Virtuel

```bash
# Avec venv
python3.12 -m venv ~/venv_sam3_ultimate
source ~/venv_sam3_ultimate/bin/activate

# OU avec conda
conda create -n sam3_env python=3.12 -y
conda activate sam3_env
```

### Étape 3: Installer PyTorch avec CUDA

```bash
# CUDA 12.8 (recommandé)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# OU CUDA 12.6
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126

# Vérifier
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Étape 4: Installer les Dépendances

```bash
pip install numpy pillow opencv-python matplotlib scipy \
    PySide6 scikit-image scikit-learn einops timm xformers

pip install git+https://github.com/huggingface/transformers.git
pip install accelerate sentencepiece protobuf huggingface-hub
```

### Étape 5: Installer SAM3

```bash
cd /tmp
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### Étape 6: Installer Depth Anything 3

```bash
cd /tmp
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
pip install -e ".[all]"
```

---

## 🔍 Différences Rocky Linux vs Ubuntu

| Composant | Ubuntu/Debian | Rocky Linux |
|-----------|---------------|-------------|
| **Gestionnaire** | `apt` | `dnf` |
| **Update** | `apt update` | `dnf check-update` |
| **Install** | `apt install` | `dnf install` |
| **Repos Extra** | PPA | EPEL + CRB |
| **Dev Tools** | `build-essential` | `Development Tools` |

---

## ⚠️ Problèmes Courants Rocky Linux

### Problème 1: Python 3.12 non disponible

**Solution**: Compiler depuis les sources (voir Étape 1 Option B)

### Problème 2: `dnf: command not found`

Sur les anciennes versions, utiliser `yum`:
```bash
# Remplacer dnf par yum
sudo yum install -y ...
```

### Problème 3: EPEL non disponible

```bash
# Rocky 9
sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm

# Rocky 8
sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
```

### Problème 4: CRB/PowerTools non trouvé

```bash
# Rocky 9 (CRB)
sudo dnf config-manager --set-enabled crb

# Rocky 8 (PowerTools)
sudo dnf config-manager --set-enabled powertools
```

### Problème 5: `python3.12-venv` non trouvé

Après compilation de Python 3.12, venv est inclus. Utiliser:
```bash
python3.12 -m venv ~/venv_sam3_ultimate
```

### Problème 6: Erreur SELinux

Si SELinux bloque l'exécution:
```bash
# Temporairement
sudo setenforce 0

# Vérifier
getenforce

# Permanent (déconseillé en production)
sudo vi /etc/selinux/config
# SELINUX=permissive
```

---

## 🧪 Test de l'Installation

```bash
# Activer l'environnement
source ~/venv_sam3_ultimate/bin/activate

# Tester
python3 test_installation.py

# Si OK, lancer l'application
python3 run.py
```

---

## 📦 Dépendances Système Rocky Linux

### Minimales

```bash
sudo dnf install -y \
    git wget curl \
    gcc gcc-c++ make \
    openssl-devel bzip2-devel libffi-devel zlib-devel \
    nvidia-driver cuda
```

### Complètes (pour compilation)

```bash
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y \
    git wget curl tar \
    gcc gcc-c++ make cmake \
    openssl-devel bzip2-devel libffi-devel zlib-devel \
    readline-devel sqlite-devel \
    xz-devel tk-devel gdbm-devel ncurses-devel \
    nvidia-driver nvidia-settings cuda
```

---

## 🔐 Firewall et SELinux

### Firewall

Si besoin d'accès réseau pour les modèles:
```bash
# Autoriser HTTP/HTTPS
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

### SELinux

Pour développement local, permissive est acceptable:
```bash
sudo setenforce 0
```

---

## 📚 Ressources Rocky Linux

- **Documentation officielle**: https://docs.rockylinux.org/
- **EPEL**: https://docs.fedoraproject.org/en-US/epel/
- **Python sur RHEL**: https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/9/html/installing_and_using_dynamic_programming_languages/assembly_installing-and-using-python_installing-and-using-dynamic-programming-languages
- **NVIDIA CUDA**: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

---

## ✅ Checklist Installation Rocky Linux

Avant de lancer l'application:

- [ ] Rocky Linux 9.5 installé
- [ ] NVIDIA GPU présent (`nvidia-smi` fonctionne)
- [ ] Pilotes NVIDIA installés (535+)
- [ ] EPEL activé
- [ ] CRB activé (Rocky 9)
- [ ] Python 3.12 installé
- [ ] Environnement virtuel créé et activé
- [ ] PyTorch avec CUDA installé
- [ ] SAM3 et DA3 installés
- [ ] HuggingFace authentifié
- [ ] Test d'installation réussi

---

## 🚀 Commandes Rapides

```bash
# Installation complète (une commande)
chmod +x install_sam3_da3_universal.sh && ./install_sam3_da3_universal.sh

# Activer environnement
source ~/venv_sam3_ultimate/bin/activate

# Authentifier HuggingFace
huggingface-cli login

# Tester
python3 test_installation.py

# Lancer
python3 run.py
```

---

## 💡 Conseils Rocky Linux

1. **Toujours utiliser `dnf`**, jamais `apt`
2. **Activer EPEL et CRB** avant d'installer des paquets
3. **Compiler Python 3.12** si non disponible dans les dépôts
4. **Vérifier SELinux** si problèmes de permissions
5. **Utiliser `sudo`** pour les installations système

---

**Version**: Compatible Rocky Linux 9.5
**Date**: 2025-12-03
**Scripts**: install_sam3_da3_universal.sh (recommandé), install_sam3_da3_rocky.sh
