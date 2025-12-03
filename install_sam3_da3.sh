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
if python3.12 --version > /dev/null 2>&1; then
    PYTHON_CMD="python3.12"
elif python3 --version | grep -q "3.12"; then
    PYTHON_CMD="python3"
else
    echo -e "${RED}✗ Python 3.12 requis mais non trouvé${NC}"
    echo "Installer Python 3.12:"
    echo "  sudo apt install python3.12 python3.12-venv"
    exit 1
fi
check_command "Python 3.12"

# 3. Créer l'environnement
echo -e "${YELLOW}[3/11] Création environnement virtuel...${NC}"
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
echo -e "${YELLOW}[4/11] Installation PyTorch 2.7.1 + CUDA 12.8...${NC}"
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
    echo -e "${RED}✗ CUDA non disponible dans PyTorch${NC}"
    echo "Essai avec CUDA 12.6..."
    pip uninstall torch torchvision torchaudio -y
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu126
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA still not available'"
    check_command "PyTorch CUDA opérationnel (12.6)"
fi

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
