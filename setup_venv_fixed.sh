#!/bin/bash

# Script d'installation VENV corrigé pour SAM3 + Depth Anything 3
# Résout les problèmes de dépendances et de compilation
# Date: 2025-12-03
# Python: 3.12
# CUDA: 12.8 or 12.6

set -e  # Arrêter en cas d'erreur

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Installation SAM3 + Depth Anything 3${NC}"
echo -e "${BLUE}Environnement Virtuel Python${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Fonction de vérification
check_success() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
        return 0
    else
        echo -e "${RED}✗ $1 - ÉCHEC${NC}"
        return 1
    fi
}

# Fonction pour afficher les erreurs et continuer
check_warning() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
    else
        echo -e "${YELLOW}⚠ $1 - Avertissement (continue)${NC}"
    fi
}

# 1. Vérifier CUDA
echo -e "${YELLOW}[1/13] Vérification CUDA...${NC}"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi > /dev/null 2>&1; then
        echo -e "${GREEN}✓ CUDA disponible${NC}"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
    else
        echo -e "${RED}✗ nvidia-smi ne fonctionne pas correctement${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ CUDA/nvidia-smi non trouvé${NC}"
    echo "Installer les pilotes NVIDIA pour utiliser le GPU"
    exit 1
fi
echo ""

# 2. Vérifier Python 3.12
echo -e "${YELLOW}[2/13] Vérification Python 3.12...${NC}"
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    PYTHON_VERSION=$(python3.12 --version)
    echo -e "${GREEN}✓ $PYTHON_VERSION disponible${NC}"
else
    echo -e "${RED}✗ Python 3.12 non trouvé${NC}"
    echo "Installer Python 3.12:"
    echo "  Ubuntu/Debian: sudo apt install python3.12 python3.12-venv python3.12-dev"
    echo "  Rocky/RHEL: Utiliser install_sam3_da3_universal.sh"
    exit 1
fi
echo ""

# 3. Définir le chemin du venv
VENV_PATH="$HOME/venv_sam3_fixed"

# 4. Nettoyer l'ancien venv si demandé
if [ -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}[3/13] Environnement existant trouvé${NC}"
    echo "Chemin: $VENV_PATH"
    read -p "Voulez-vous le supprimer et recréer? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Suppression de l'ancien environnement..."
        rm -rf "$VENV_PATH"
        echo -e "${GREEN}✓ Ancien environnement supprimé${NC}"
    else
        echo "Conservation de l'environnement existant"
    fi
else
    echo -e "${YELLOW}[3/13] Pas d'environnement existant${NC}"
fi
echo ""

# 5. Créer le venv
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}[4/13] Création de l'environnement virtuel...${NC}"
    $PYTHON_CMD -m venv "$VENV_PATH"
    check_success "Environnement virtuel créé"
else
    echo -e "${YELLOW}[4/13] Utilisation de l'environnement existant${NC}"
fi
echo ""

# 6. Activer le venv
echo -e "${YELLOW}[5/13] Activation de l'environnement...${NC}"
source "$VENV_PATH/bin/activate"

# Vérifier l'activation
CURRENT_PYTHON=$(which python3)
if [[ "$CURRENT_PYTHON" == *"venv_sam3_fixed"* ]]; then
    echo -e "${GREEN}✓ Environnement activé: $CURRENT_PYTHON${NC}"
else
    echo -e "${RED}✗ Erreur d'activation${NC}"
    echo "Python actuel: $CURRENT_PYTHON"
    exit 1
fi
echo ""

# 7. Mettre à jour pip, setuptools, wheel
echo -e "${YELLOW}[6/13] Mise à jour des outils de base...${NC}"
pip install --upgrade pip setuptools wheel
check_success "Outils de base mis à jour"
echo ""

# 8. Installer PyTorch AVANT tout le reste (CRITIQUE)
echo -e "${YELLOW}[7/13] Installation PyTorch 2.7.1 + CUDA 12.8...${NC}"
echo "Ceci peut prendre quelques minutes..."
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
check_success "PyTorch installé"

# Vérifier PyTorch CUDA
echo "Vérification de CUDA avec PyTorch..."
if python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    echo -e "${GREEN}✓ PyTorch CUDA opérationnel${NC}"
    python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}'); print(f'  Devices: {torch.cuda.device_count()}')"
else
    echo -e "${YELLOW}⚠ CUDA 12.8 non disponible, essai avec CUDA 12.6...${NC}"
    pip uninstall torch torchvision torchaudio -y
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu126

    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo -e "${GREEN}✓ PyTorch CUDA 12.6 opérationnel${NC}"
    else
        echo -e "${RED}✗ CUDA toujours non disponible${NC}"
        echo "Vérifiez votre installation CUDA et pilotes NVIDIA"
        exit 1
    fi
fi
echo ""

# 9. Installer les dépendances de base
echo -e "${YELLOW}[8/13] Installation des bibliothèques de base...${NC}"
pip install \
    numpy>=1.26.0 \
    pillow>=10.0.0 \
    opencv-python>=4.8.0 \
    scipy>=1.11.0 \
    matplotlib>=3.7.0 \
    scikit-image>=0.21.0 \
    scikit-learn>=1.3.0 \
    PySide6>=6.5.0
check_success "Bibliothèques de base installées"
echo ""

# 10. Installer les utilitaires ML (einops, timm) AVANT xformers
echo -e "${YELLOW}[9/13] Installation des utilitaires ML...${NC}"
pip install einops>=0.7.0 timm>=0.9.0
check_success "Utilitaires ML installés"
echo ""

# 11. Installer xformers (nécessite torch déjà installé)
echo -e "${YELLOW}[10/13] Installation xformers...${NC}"
echo "Compilation de xformers (peut prendre 5-10 minutes)..."
pip install xformers --no-build-isolation
check_warning "xformers installé"
echo ""

# 12. Installer Hugging Face et Transformers
echo -e "${YELLOW}[11/13] Installation Hugging Face et Transformers...${NC}"
pip install \
    huggingface-hub>=0.20.0 \
    accelerate>=0.25.0 \
    sentencepiece>=0.1.99 \
    protobuf>=4.25.0

# Installer Transformers depuis main (requis pour SAM3)
echo "Installation de Transformers (main branch)..."
pip install git+https://github.com/huggingface/transformers.git
check_success "Transformers installé"
echo ""

# 13. Installer SAM3 depuis GitHub
echo -e "${YELLOW}[12/13] Installation SAM3...${NC}"
cd /tmp
if [ -d "sam3" ]; then
    echo "Suppression de l'ancien clone SAM3..."
    rm -rf sam3
fi

echo "Clonage de SAM3..."
git clone https://github.com/facebookresearch/sam3.git
cd sam3
echo "Installation de SAM3 en mode éditable..."
pip install -e .
check_success "SAM3 installé"
cd - > /dev/null
echo ""

# 14. Installer Depth Anything 3 depuis GitHub
echo -e "${YELLOW}[13/13] Installation Depth Anything 3...${NC}"
cd /tmp
if [ -d "Depth-Anything-3" ]; then
    echo "Suppression de l'ancien clone Depth-Anything-3..."
    rm -rf Depth-Anything-3
fi

echo "Clonage de Depth Anything 3..."
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3

# Installer SANS xformers car déjà installé
echo "Installation de Depth Anything 3 (mode éditable)..."
# On installe d'abord sans les extras pour éviter les conflits
pip install -e . --no-deps
# Puis on installe les dépendances manquantes
pip install evo pycocotools decord pre-commit || true

check_warning "Depth Anything 3 installé"
cd - > /dev/null
echo ""

# 15. Installer les outils de test
echo -e "${YELLOW}Installation des outils de test...${NC}"
pip install pytest>=7.4.0 pytest-cov>=4.1.0
check_success "Outils de test installés"
echo ""

# 16. Vérification finale
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}VÉRIFICATION DE L'INSTALLATION${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${YELLOW}=== Versions Installées ===${NC}"
python3 -c "import sys; print(f'Python: {sys.version}')"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python3 -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU(s): {torch.cuda.device_count()}')" 2>/dev/null || echo "GPU count: 0"
python3 -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"
python3 -c "import xformers; print(f'xformers: {xformers.__version__}')" 2>/dev/null || echo "xformers: Non installé"
python3 -c "import numpy; print(f'numpy: {numpy.__version__}')"
python3 -c "import cv2; print(f'opencv: {cv2.__version__}')"
python3 -c "import PySide6; print(f'PySide6: OK')" 2>/dev/null || echo "PySide6: Non installé"
echo ""

echo -e "${YELLOW}=== Test des Imports ===${NC}"
python3 -c "from sam3.model_builder import build_sam3_image_model; print('✓ SAM3 (GitHub)')" 2>/dev/null || echo "✗ SAM3 - Erreur d'import"
python3 -c "from depth_anything_3.api import DepthAnything3; print('✓ Depth Anything 3 API')" 2>/dev/null || echo "⚠ Depth Anything 3 - Import partiel"
python3 -c "import depth_anything_3; print('✓ Depth Anything 3 module')" 2>/dev/null || echo "✗ Depth Anything 3 - Erreur d'import"
echo ""

# 17. Afficher les instructions finales
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ INSTALLATION TERMINÉE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Pour activer l'environnement:${NC}"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo -e "${YELLOW}Configuration HuggingFace (REQUIS pour SAM3):${NC}"
echo "  1. Obtenir un token: https://huggingface.co/settings/tokens"
echo "  2. Se connecter: huggingface-cli login"
echo "  3. Demander l'accès à SAM3: https://huggingface.co/facebook/sam3"
echo ""
echo -e "${YELLOW}Tester l'installation:${NC}"
echo "  python3 test_installation.py"
echo ""
echo -e "${YELLOW}Lancer l'application:${NC}"
echo "  cd /home/user/sam4"
echo "  python3 run.py"
echo ""
echo -e "${BLUE}Chemin du venv: $VENV_PATH${NC}"
echo ""

# Créer un script d'activation rapide
cat > /home/user/sam4/activate_venv.sh << 'EOF'
#!/bin/bash
# Script d'activation rapide du venv SAM3
source ~/venv_sam3_fixed/bin/activate
echo "Environnement SAM3 activé!"
echo "Python: $(which python3)"
echo "Version: $(python3 --version)"
EOF
chmod +x /home/user/sam4/activate_venv.sh

echo -e "${GREEN}Script d'activation rapide créé: ./activate_venv.sh${NC}"
echo ""
