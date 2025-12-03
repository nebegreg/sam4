#!/bin/bash

# Installation UNIVERSELLE pour SAM3 + Depth Anything 3
# Compatible: Ubuntu, Debian, Rocky Linux, RHEL, CentOS, AlmaLinux, Fedora
# Détection automatique de la distribution
# Date: 2025-12-03
# Python 3.12, PyTorch 2.7.1, CUDA 12.8/12.6

set -e  # Arrêter en cas d'erreur

echo "=== Installation UNIVERSELLE SAM3 + Depth Anything 3 ==="
echo ""

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Détecter la distribution Linux
echo -e "${BLUE}[INFO] Détection du système...${NC}"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
    VERSION_ID=$VERSION_ID
    echo "Distribution: $NAME $VERSION"
else
    echo -e "${RED}✗ Impossible de détecter la distribution${NC}"
    exit 1
fi

# Déterminer le gestionnaire de paquets
case "$DISTRO" in
    ubuntu|debian|linuxmint|pop)
        PKG_MANAGER="apt"
        PKG_UPDATE="sudo apt update"
        PKG_INSTALL="sudo apt install -y"
        PYTHON_PKG="python3.12 python3.12-venv python3.12-dev"
        ;;
    rocky|rhel|centos|almalinux|fedora)
        PKG_MANAGER="dnf"
        PKG_UPDATE="sudo dnf check-update || true"
        PKG_INSTALL="sudo dnf install -y"
        PYTHON_PKG="python3.12 python3.12-devel python3.12-pip"
        ;;
    arch|manjaro)
        PKG_MANAGER="pacman"
        PKG_UPDATE="sudo pacman -Sy"
        PKG_INSTALL="sudo pacman -S --noconfirm"
        PYTHON_PKG="python"
        ;;
    opensuse*|sles)
        PKG_MANAGER="zypper"
        PKG_UPDATE="sudo zypper refresh"
        PKG_INSTALL="sudo zypper install -y"
        PYTHON_PKG="python312 python312-devel python312-pip"
        ;;
    *)
        echo -e "${RED}✗ Distribution '$DISTRO' non supportée automatiquement${NC}"
        echo "Distributions supportées: Ubuntu, Debian, Rocky Linux, RHEL, CentOS, AlmaLinux, Fedora, Arch, openSUSE"
        echo ""
        echo "Pour installer manuellement:"
        echo "1. Installer Python 3.12"
        echo "2. Lancer: ./install_sam3_da3_rocky.sh (pour RHEL-based)"
        echo "   ou      ./install_sam3_da3.sh (pour Debian-based)"
        exit 1
        ;;
esac

echo -e "${GREEN}✓ Gestionnaire de paquets: $PKG_MANAGER${NC}"
echo ""

# 1. Vérifier CUDA
echo -e "${YELLOW}[1/12] Vérification CUDA...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi > /dev/null 2>&1
    check_command "CUDA disponible"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
else
    echo -e "${RED}✗ CUDA/nvidia-smi non trouvé${NC}"
    echo "Installer les pilotes NVIDIA:"
    case "$PKG_MANAGER" in
        apt)
            echo "  sudo apt install -y nvidia-driver-535"
            ;;
        dnf)
            echo "  sudo dnf install -y nvidia-driver nvidia-settings cuda"
            ;;
        *)
            echo "  Consulter la documentation de votre distribution"
            ;;
    esac
    exit 1
fi

# 2. Vérifier/Installer Python 3.12
echo -e "${YELLOW}[2/12] Vérification Python 3.12...${NC}"
if python3.12 --version > /dev/null 2>&1; then
    PYTHON_CMD="python3.12"
    echo -e "${GREEN}✓ Python 3.12 déjà installé${NC}"
elif python3 --version 2>&1 | grep -q "3.12"; then
    PYTHON_CMD="python3"
    echo -e "${GREEN}✓ Python 3.12 déjà installé (python3)${NC}"
else
    echo -e "${YELLOW}Python 3.12 non trouvé, installation...${NC}"

    case "$PKG_MANAGER" in
        apt)
            # Pour Ubuntu/Debian
            $PKG_UPDATE
            if ! $PKG_INSTALL $PYTHON_PKG 2>/dev/null; then
                # Essayer avec deadsnakes PPA
                echo "Ajout du PPA deadsnakes pour Python 3.12..."
                $PKG_INSTALL software-properties-common
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                $PKG_UPDATE
                $PKG_INSTALL $PYTHON_PKG
            fi
            PYTHON_CMD="python3.12"
            ;;

        dnf)
            # Pour Rocky/RHEL/CentOS/Fedora
            $PKG_UPDATE

            # Activer EPEL si nécessaire
            if ! dnf repolist enabled | grep -q epel; then
                echo "Activation d'EPEL..."
                $PKG_INSTALL epel-release
            fi

            # Essayer d'installer depuis les dépôts
            if ! $PKG_INSTALL $PYTHON_PKG 2>/dev/null; then
                echo -e "${YELLOW}Compilation de Python 3.12 depuis les sources...${NC}"

                # Activer CRB/PowerTools
                sudo dnf config-manager --set-enabled crb 2>/dev/null || \
                    sudo dnf config-manager --set-enabled powertools 2>/dev/null || true

                # Installer les dépendances de compilation
                $PKG_INSTALL gcc gcc-c++ make openssl-devel bzip2-devel \
                    libffi-devel zlib-devel wget tar

                # Compiler Python 3.12
                cd /tmp
                PYTHON_VERSION="3.12.7"
                if [ ! -f "Python-${PYTHON_VERSION}.tgz" ]; then
                    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
                fi
                tar xzf Python-${PYTHON_VERSION}.tgz
                cd Python-${PYTHON_VERSION}
                ./configure --enable-optimizations --with-ensurepip=install
                make -j $(nproc)
                sudo make altinstall
                cd /tmp
                rm -rf Python-${PYTHON_VERSION}*
            fi
            PYTHON_CMD="python3.12"
            ;;

        pacman)
            # Pour Arch Linux
            $PKG_UPDATE
            $PKG_INSTALL $PYTHON_PKG
            PYTHON_CMD="python3"
            ;;

        zypper)
            # Pour openSUSE
            $PKG_UPDATE
            $PKG_INSTALL $PYTHON_PKG
            PYTHON_CMD="python3.12"
            ;;
    esac

    check_command "Python 3.12 installé"
fi

# Vérifier que Python 3.12 fonctionne
$PYTHON_CMD --version
check_command "Python 3.12 fonctionnel"

# 3. Créer l'environnement virtuel
echo -e "${YELLOW}[3/12] Création environnement virtuel...${NC}"
if command -v conda &> /dev/null; then
    echo "Conda détecté, création environnement conda..."
    conda create -n sam3_env python=3.12 -y
    eval "$(conda shell.bash hook)"
    conda activate sam3_env
    check_command "Environnement conda créé"
else
    echo "Création environnement venv..."
    $PYTHON_CMD -m venv ~/venv_sam3_ultimate
    source ~/venv_sam3_ultimate/bin/activate
    check_command "Environnement venv créé"
fi

# Vérifier que l'environnement est activé
PYTHON_PATH=$(which python3)
if [[ "$PYTHON_PATH" == *"venv_sam3_ultimate"* ]] || [[ "$PYTHON_PATH" == *"sam3_env"* ]]; then
    echo -e "${GREEN}✓ Environnement activé: $PYTHON_PATH${NC}"
else
    echo -e "${RED}✗ Erreur: environnement pas activé correctement${NC}"
    echo "Python path: $PYTHON_PATH"
    exit 1
fi

# 4. Installer PyTorch 2.7.1 avec CUDA 12.8
echo -e "${YELLOW}[4/12] Installation PyTorch 2.7.1 + CUDA 12.8...${NC}"
pip install --upgrade pip setuptools wheel
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
check_command "PyTorch installé"

# Vérifier PyTorch CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PyTorch CUDA opérationnel${NC}"
    python3 -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
else
    echo -e "${YELLOW}⚠ CUDA 12.8 non disponible, essai avec CUDA 12.6...${NC}"
    pip uninstall torch torchvision torchaudio -y
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu126
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA still not available'"
    check_command "PyTorch CUDA opérationnel (12.6)"
fi

# 5. Installer dépendances de base
echo -e "${YELLOW}[5/12] Installation dépendances de base...${NC}"
pip install numpy pillow opencv-python matplotlib scipy \
    PySide6 scikit-image scikit-learn einops timm
check_command "Dépendances de base installées"

# 6. Installer xformers
echo -e "${YELLOW}[6/12] Installation xformers...${NC}"
pip install xformers
check_command "xformers installé"

# 7. Installer Transformers (main branch)
echo -e "${YELLOW}[7/12] Installation Transformers (main)...${NC}"
pip install git+https://github.com/huggingface/transformers.git
pip install accelerate sentencepiece protobuf huggingface-hub
check_command "Transformers installé"

# 8. Installer SAM3
echo -e "${YELLOW}[8/12] Installation SAM3...${NC}"
cd /tmp
if [ -d "sam3" ]; then rm -rf sam3; fi
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
check_command "SAM3 installé"

# 9. Installer Depth Anything 3
echo -e "${YELLOW}[9/12] Installation Depth Anything 3...${NC}"
cd /tmp
if [ -d "Depth-Anything-3" ]; then rm -rf Depth-Anything-3; fi
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
pip install -e ".[all]"
check_command "Depth Anything 3 installé"

# 10. Installer pytest
echo -e "${YELLOW}[10/12] Installation outils de test...${NC}"
pip install pytest pytest-cov
check_command "Pytest installé"

# 11. Vérification finale
echo -e "${YELLOW}[11/12] Vérification finale...${NC}"
echo ""
echo "=== Versions Installées ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python3 -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
python3 -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"
python3 -c "import xformers; print(f'xformers: {xformers.__version__}')"
python3 -c "import numpy; print(f'numpy: {numpy.__version__}')"
python3 -c "import PySide6; print('PySide6: OK')"
python3 -c "import cv2; print(f'opencv: {cv2.__version__}')"
echo ""

# Test imports
echo "=== Test des Imports ==="
python3 -c "from transformers import Sam3Model, Sam3Processor; print('✓ SAM3 (transformers)')" 2>/dev/null || echo "⚠ SAM3 (transformers) - Utiliser sam3 GitHub à la place"
python3 -c "from sam3.model_builder import build_sam3_image_model; print('✓ SAM3 (GitHub repo)')"
python3 -c "from depth_anything_3.api import DepthAnything3; print('✓ Depth Anything 3')"
echo ""

echo -e "${GREEN}=== ✓ Installation Terminée avec Succès! ===${NC}"
echo ""
echo "Distribution: $NAME"
echo "Gestionnaire de paquets: $PKG_MANAGER"
echo ""
echo "Pour activer l'environnement:"
if command -v conda &> /dev/null && [[ "$PYTHON_PATH" == *"sam3_env"* ]]; then
    echo "  conda activate sam3_env"
else
    echo "  source ~/venv_sam3_ultimate/bin/activate"
fi
echo ""
echo -e "${YELLOW}IMPORTANT: Authentification HuggingFace requise pour SAM3:${NC}"
echo "  huggingface-cli login"
echo "  Token: https://huggingface.co/settings/tokens"
echo "  Demander accès: https://huggingface.co/facebook/sam3"
echo ""
echo "Prochaines étapes:"
echo "  1. Se connecter à HuggingFace: huggingface-cli login"
echo "  2. Tester l'installation: python3 test_installation.py"
echo "  3. Lancer l'application: python3 run.py"
echo ""
