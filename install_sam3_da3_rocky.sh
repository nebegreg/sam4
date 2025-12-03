#!/bin/bash

# Installation automatique pour SAM3 + Depth Anything 3
# Compatible: Rocky Linux, RHEL, CentOS, AlmaLinux, Fedora
# Date: 2025-12-03
# Python 3.12, PyTorch 2.7.1, CUDA 12.8/12.6

set -e  # Arrêter en cas d'erreur

echo "=== Installation SAM3 + Depth Anything 3 (Rocky Linux) ==="
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

# Détecter la version de Rocky Linux
echo -e "${YELLOW}[INFO] Détection du système...${NC}"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "Distribution: $NAME $VERSION"
else
    echo -e "${RED}✗ Impossible de détecter la distribution${NC}"
    exit 1
fi

# 1. Vérifier CUDA
echo -e "${YELLOW}[1/12] Vérification CUDA...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi > /dev/null 2>&1
    check_command "CUDA disponible"
else
    echo -e "${RED}✗ CUDA/nvidia-smi non trouvé${NC}"
    echo "Installer les pilotes NVIDIA:"
    echo "  sudo dnf install -y nvidia-driver nvidia-settings cuda"
    exit 1
fi

# 2. Vérifier Python 3.12
echo -e "${YELLOW}[2/12] Vérification Python 3.12...${NC}"
if python3.12 --version > /dev/null 2>&1; then
    PYTHON_CMD="python3.12"
    echo -e "${GREEN}✓ Python 3.12 déjà installé${NC}"
elif python3 --version 2>&1 | grep -q "3.12"; then
    PYTHON_CMD="python3"
    echo -e "${GREEN}✓ Python 3.12 déjà installé${NC}"
else
    echo -e "${YELLOW}Python 3.12 non trouvé, installation...${NC}"

    # Vérifier si EPEL est activé
    if ! dnf repolist enabled | grep -q epel; then
        echo "Activation d'EPEL..."
        sudo dnf install -y epel-release
    fi

    # Pour Rocky Linux 9, Python 3.12 peut être dans les dépôts ou via CRB
    echo "Tentative d'installation de Python 3.12..."

    # Option 1: Essayer depuis les dépôts standards
    if sudo dnf install -y python3.12 python3.12-devel python3.12-pip 2>/dev/null; then
        PYTHON_CMD="python3.12"
        check_command "Python 3.12 installé depuis dnf"
    else
        # Option 2: Depuis les sources ou alternative
        echo -e "${YELLOW}Python 3.12 non disponible dans les dépôts${NC}"
        echo -e "${YELLOW}Installation depuis le repo Python community...${NC}"

        # Activer CRB (CodeReady Builder) pour Rocky 9
        sudo dnf config-manager --set-enabled crb 2>/dev/null || sudo dnf config-manager --set-enabled powertools

        # Installer les dépendances de compilation
        sudo dnf groupinstall -y "Development Tools"
        sudo dnf install -y openssl-devel bzip2-devel libffi-devel zlib-devel wget

        # Télécharger et compiler Python 3.12
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

        PYTHON_CMD="python3.12"
        cd /tmp
        rm -rf Python-${PYTHON_VERSION}*
        check_command "Python 3.12 compilé et installé"
    fi
fi

# Vérifier que Python 3.12 est bien installé
$PYTHON_CMD --version
check_command "Python 3.12 fonctionnel"

# 3. Créer l'environnement
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
