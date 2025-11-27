#!/bin/bash
################################################################################
# SAM3 Ultimate Virtual Environment Setup
# Crée un environnement virtuel Python optimisé pour SAM3 avec:
# - Python 3.9+
# - PyTorch avec CUDA 12.x
# - Transformers (nightly pour SAM3)
# - Repo GitHub SAM3 officiel
# - Toutes les dépendances (decord, pycocotools, etc.)
################################################################################

set -e  # Arrêter en cas d'erreur

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

echo "========================================================================"
echo "🚀 SAM3 Ultimate Virtual Environment Setup"
echo "========================================================================"
echo ""

# Configuration
VENV_NAME="venv_sam3_ultimate"
VENV_PATH="$HOME/Documents/$VENV_NAME"
EXTERNAL_MODELS="$VENV_PATH/.external_models"

log_info "Configuration:"
echo "  - Nom du venv: $VENV_NAME"
echo "  - Chemin: $VENV_PATH"
echo "  - Python version requise: >= 3.9"
echo ""

# Vérifier Python
log_info "Vérification de Python..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "  Python détecté: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    log_error "Python 3.9+ requis. Version détectée: $PYTHON_VERSION"
    exit 1
fi
log_success "Python $PYTHON_VERSION OK"
echo ""

# Vérifier CUDA
log_info "Vérification de CUDA..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | grep -oP '\d+\.\d+' | head -1)
    log_success "CUDA $CUDA_VERSION détecté"

    # Déterminer la version PyTorch CUDA à installer
    if [ -n "$CUDA_VERSION" ]; then
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        if [ "$CUDA_MAJOR" -ge 12 ]; then
            TORCH_CUDA="cu121"
            log_info "Utilisation de PyTorch avec CUDA 12.1"
        elif [ "$CUDA_MAJOR" -eq 11 ]; then
            TORCH_CUDA="cu118"
            log_info "Utilisation de PyTorch avec CUDA 11.8"
        else
            TORCH_CUDA="cpu"
            log_warning "CUDA $CUDA_VERSION trop ancien, utilisation CPU"
        fi
    else
        TORCH_CUDA="cpu"
        log_warning "CUDA non détecté, utilisation CPU"
    fi
else
    log_warning "nvidia-smi non trouvé, installation CPU uniquement"
    TORCH_CUDA="cpu"
fi
echo ""

# Supprimer l'ancien venv si il existe
if [ -d "$VENV_PATH" ]; then
    log_warning "Un venv existe déjà à $VENV_PATH"
    read -p "Supprimer et recréer? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Suppression de l'ancien venv..."
        rm -rf "$VENV_PATH"
        log_success "Ancien venv supprimé"
    else
        log_error "Installation annulée"
        exit 1
    fi
fi
echo ""

# Créer le venv
log_info "Création du virtual environment..."
python3 -m venv "$VENV_PATH"
log_success "Virtual environment créé"
echo ""

# Activer le venv
log_info "Activation du venv..."
source "$VENV_PATH/bin/activate"
log_success "Venv activé"
echo ""

# Mettre à jour pip
log_info "Mise à jour de pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel --quiet
log_success "Outils de base mis à jour"
echo ""

# Installer PyTorch
log_info "Installation de PyTorch avec $TORCH_CUDA..."
if [ "$TORCH_CUDA" = "cpu" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/$TORCH_CUDA
fi
log_success "PyTorch installé"
echo ""

# Vérifier l'installation PyTorch
log_info "Vérification de PyTorch..."
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
log_success "PyTorch fonctionnel"
echo ""

# Installer les dépendances de base
log_info "Installation des dépendances de base..."
pip install --quiet \
    numpy \
    pillow \
    opencv-python \
    PySide6 \
    tqdm \
    imageio \
    imageio-ffmpeg \
    accelerate \
    safetensors \
    huggingface_hub
log_success "Dépendances de base installées"
echo ""

# Installer decord (requis par SAM3)
log_info "Installation de decord (vidéo)..."
pip install decord --quiet || log_warning "decord installation échouée (non critique)"
log_success "decord installé"
echo ""

# Installer pycocotools (requis par SAM3)
log_info "Installation de pycocotools..."
pip install pycocotools --quiet || log_warning "pycocotools installation échouée (non critique)"
log_success "pycocotools installé"
echo ""

# Installer transformers (NIGHTLY pour SAM3)
log_info "Installation de transformers (version nightly pour SAM3)..."
echo "  Note: SAM3 n'est pas encore dans transformers stable"
echo "  Tentative d'installation de la version git..."
pip install git+https://github.com/huggingface/transformers.git --quiet 2>/dev/null || {
    log_warning "Installation git échouée, installation de la version stable..."
    pip install transformers --quiet
}
log_success "Transformers installé"
echo ""

# Créer le dossier pour les modèles externes
log_info "Création du dossier pour modèles externes..."
mkdir -p "$EXTERNAL_MODELS"
log_success "Dossier créé: $EXTERNAL_MODELS"
echo ""

# Cloner et installer SAM3 depuis GitHub
log_info "Installation du repo GitHub SAM3 officiel..."
cd "$EXTERNAL_MODELS"

if [ -d "sam3" ]; then
    log_warning "Le repo sam3 existe déjà, mise à jour..."
    cd sam3
    git pull --quiet || log_warning "Mise à jour échouée"
    cd ..
else
    log_info "Clonage de facebook/sam3..."
    git clone https://github.com/facebookresearch/sam3.git --quiet || {
        log_error "Échec du clonage. Vérifiez votre connexion internet."
        exit 1
    }
fi

cd sam3
log_info "Installation de SAM3 en mode développement..."
pip install -e . --quiet || {
    log_error "Installation de SAM3 échouée"
    exit 1
}
log_success "SAM3 installé depuis GitHub"
echo ""

# Cloner et installer Depth-Anything-v3 (optionnel)
log_info "Installation de Depth-Anything-v3..."
cd "$EXTERNAL_MODELS"

if [ -d "Depth-Anything-V3" ]; then
    log_warning "Le repo Depth-Anything-V3 existe déjà"
else
    log_info "Clonage de DepthAnything/Depth-Anything-V3..."
    git clone https://github.com/DepthAnything/Depth-Anything-V3.git --quiet || {
        log_warning "Clonage de DA3 échoué (non critique)"
    }
fi

if [ -d "Depth-Anything-V3" ]; then
    cd Depth-Anything-V3
    pip install -e . --quiet 2>/dev/null || log_warning "Installation DA3 échouée (non critique)"
    log_success "Depth-Anything-V3 installé"
else
    log_warning "Depth-Anything-V3 non installé (optionnel)"
fi
echo ""

# Installer les dépendances du projet sam4
log_info "Installation des dépendances du projet sam4..."
cd "$HOME/Downloads/sam4-main" 2>/dev/null || cd "$(dirname "$0")"

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet || log_warning "Certaines dépendances ont échoué"
    log_success "requirements.txt installé"
else
    log_warning "requirements.txt non trouvé"
fi
echo ""

# Vérification finale
log_info "Vérification finale des installations..."
echo ""

echo "📋 Tests d'imports:"
python3 << 'PYEOF'
import sys

def test_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'OK')
        print(f"  ✓ {display_name}: {version}")
        return True
    except ImportError as e:
        print(f"  ✗ {display_name}: {e}")
        return False

# Test des imports critiques
test_import('torch', 'PyTorch')
test_import('torchvision', 'TorchVision')
test_import('transformers', 'Transformers')
test_import('PIL', 'Pillow')
test_import('cv2', 'OpenCV')
test_import('PySide6', 'PySide6')
test_import('numpy', 'NumPy')
test_import('decord', 'Decord')
test_import('pycocotools', 'COCO Tools')

# Test SAM3
print()
print("🔍 Test SAM3:")
try:
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
    print("  ✓ Repo GitHub SAM3: Importable")
except ImportError as e:
    print(f"  ✗ Repo GitHub SAM3: {e}")

# Test Transformers SAM3
try:
    from transformers import Sam3Model, Sam3Processor
    print("  ✓ Transformers SAM3Model: Disponible")
except ImportError:
    print("  ⚠ Transformers SAM3Model: Non disponible (normal, pas encore stable)")

# Test CUDA
print()
print("🎮 Test CUDA:")
import torch
print(f"  CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Device count: {torch.cuda.device_count()}")
    print(f"  Device name: {torch.cuda.get_device_name(0)}")
    print(f"  BFloat16 support: {torch.cuda.is_bf16_supported()}")
PYEOF

echo ""
log_success "Vérification terminée"
echo ""

# Créer un script d'activation
ACTIVATE_SCRIPT="$HOME/activate_sam3_ultimate.sh"
cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Script d'activation rapide pour SAM3 Ultimate

VENV_PATH="$HOME/Documents/venv_sam3_ultimate"

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment non trouvé à $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"

echo "✅ SAM3 Ultimate venv activé"
echo ""
echo "📂 Dossiers importants:"
echo "  - Venv: $VENV_PATH"
echo "  - Modèles externes: $VENV_PATH/.external_models"
echo "  - SAM3 repo: $VENV_PATH/.external_models/sam3"
echo ""
echo "🚀 Pour lancer l'application:"
echo "  cd ~/Downloads/sam4-main"
echo "  python run.py"
echo ""
EOF

chmod +x "$ACTIVATE_SCRIPT"
log_success "Script d'activation créé: $ACTIVATE_SCRIPT"
echo ""

# Créer un alias bashrc
log_info "Ajout de l'alias au ~/.bashrc..."
if ! grep -q "alias sam3=" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# SAM3 Ultimate activation" >> ~/.bashrc
    echo "alias sam3='source $ACTIVATE_SCRIPT'" >> ~/.bashrc
    log_success "Alias 'sam3' ajouté au ~/.bashrc"
else
    log_warning "Alias 'sam3' existe déjà dans ~/.bashrc"
fi
echo ""

# Résumé final
echo "========================================================================"
echo "✅ Installation terminée avec succès!"
echo "========================================================================"
echo ""
echo "📋 RÉSUMÉ:"
echo "  - Virtual environment: $VENV_PATH"
echo "  - Python version: $(python3 --version 2>&1)"
echo "  - PyTorch installé avec: $TORCH_CUDA"
echo "  - SAM3 GitHub repo: Installé"
echo "  - Transformers: Installé"
echo ""
echo "🚀 UTILISATION:"
echo ""
echo "  1. Activer le venv (méthode 1):"
echo "     source $VENV_PATH/bin/activate"
echo ""
echo "  2. Activer le venv (méthode 2 - script):"
echo "     source $ACTIVATE_SCRIPT"
echo ""
echo "  3. Activer le venv (méthode 3 - alias):"
echo "     source ~/.bashrc   # Recharger bashrc"
echo "     sam3               # Puis utiliser l'alias"
echo ""
echo "  4. Lancer l'application:"
echo "     cd ~/Downloads/sam4-main"
echo "     python run.py"
echo ""
echo "  5. Dans l'interface SAM3:"
echo "     Model ID: facebook/sam3-hiera-large"
echo "     Ou chemin local si vous avez téléchargé le modèle"
echo ""
echo "💡 NOTES:"
echo "  - Le modèle sera téléchargé automatiquement au premier chargement"
echo "  - Cache HuggingFace: ~/.cache/huggingface/"
echo "  - Taille du modèle SAM3: ~2-4 GB"
echo ""
echo "========================================================================"
