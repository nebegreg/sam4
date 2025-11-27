#!/bin/bash
# Script d'installation complet pour SAM3 Roto Ultimate
# Crée un environnement virtuel dans ~/Documents/venv_sam avec toutes les dépendances

set -e  # Arrêter en cas d'erreur

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo "Installation SAM3 Roto Ultimate"
echo -e "========================================${NC}"

# Variables
VENV_PATH="$HOME/Documents/venv_sam"
PYTHON_MIN_VERSION="3.12"
MODELS_DIR="$HOME/Documents/venv_sam/.external_models"

# Fonction pour afficher les messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Vérifier Python
log_info "Vérification de Python..."
if ! command -v python3 &> /dev/null; then
    log_error "Python3 n'est pas installé"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
log_success "Python $PYTHON_VERSION détecté"

if (( $(echo "$PYTHON_VERSION < $PYTHON_MIN_VERSION" | bc -l) )); then
    log_warning "Python $PYTHON_MIN_VERSION+ recommandé, vous avez $PYTHON_VERSION"
    read -p "Continuer quand même ? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Vérifier Git
log_info "Vérification de Git..."
if ! command -v git &> /dev/null; then
    log_error "Git n'est pas installé. Installez-le avec: sudo yum install git"
    exit 1
fi
log_success "Git détecté"

# Créer le dossier Documents si nécessaire
if [ ! -d "$HOME/Documents" ]; then
    log_info "Création du dossier Documents..."
    mkdir -p "$HOME/Documents"
fi

# Supprimer l'ancien venv s'il existe
if [ -d "$VENV_PATH" ]; then
    log_warning "Un environnement virtuel existe déjà à $VENV_PATH"
    read -p "Voulez-vous le supprimer et recommencer ? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Suppression de l'ancien environnement..."
        rm -rf "$VENV_PATH"
        log_success "Ancien environnement supprimé"
    else
        log_info "Utilisation de l'environnement existant"
    fi
fi

# Créer le venv
if [ ! -d "$VENV_PATH" ]; then
    log_info "Création de l'environnement virtuel dans $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
    log_success "Environnement virtuel créé"
else
    log_success "Environnement virtuel trouvé"
fi

# Activer le venv
log_info "Activation de l'environnement virtuel..."
source "$VENV_PATH/bin/activate"
log_success "Environnement activé"

# Mettre à jour pip, wheel, setuptools
log_info "Mise à jour de pip, wheel et setuptools..."
pip install --upgrade pip wheel setuptools --quiet
log_success "Outils de base mis à jour"

# Installer PyTorch (compatible CUDA si disponible)
log_info "Installation de PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    log_info "GPU NVIDIA détecté, installation de PyTorch avec support CUDA..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
else
    log_warning "Pas de GPU NVIDIA détecté, installation de PyTorch CPU..."
    pip install torch torchvision --quiet
fi
log_success "PyTorch installé"

# Aller dans le répertoire du script (ou rester où on est)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Installer les dépendances de base
if [ -f "requirements.txt" ]; then
    log_info "Installation des dépendances depuis requirements.txt..."
    pip install -r requirements.txt --quiet
    log_success "Dépendances de base installées"
else
    log_warning "requirements.txt non trouvé, installation manuelle..."
    pip install numpy pillow opencv-python PySide6 tqdm imageio imageio-ffmpeg accelerate decord --quiet
    log_success "Dépendances de base installées"
fi

# Installer decord séparément si l'installation depuis requirements a échoué
log_info "Vérification de decord (requis par SAM3)..."
if ! python3 -c "import decord" 2>/dev/null; then
    log_warning "Decord non trouvé, installation..."
    pip install decord --quiet || log_warning "Decord n'a pas pu être installé automatiquement"
else
    log_success "Decord détecté"
fi

# Installer pycocotools séparément si nécessaire
log_info "Vérification de pycocotools (requis par SAM3)..."
if ! python3 -c "import pycocotools" 2>/dev/null; then
    log_warning "Pycocotools non trouvé, installation..."
    pip install pycocotools --quiet || log_warning "Pycocotools n'a pas pu être installé automatiquement"
else
    log_success "Pycocotools détecté"
fi

# Installer les dépendances optionnelles
if [ -f "requirements_optional.txt" ]; then
    log_info "Installation des dépendances optionnelles (opencv-contrib)..."
    pip install -r requirements_optional.txt --quiet || log_warning "Certaines dépendances optionnelles n'ont pas pu être installées"
    log_success "Dépendances optionnelles traitées"
fi

# Créer le dossier pour les modèles
log_info "Création du dossier pour les modèles externes..."
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# Installer SAM3
log_info "Installation de SAM3 depuis GitHub..."
if [ -d "sam3" ]; then
    log_warning "SAM3 déjà cloné, mise à jour..."
    cd sam3
    git pull --quiet
    cd ..
else
    log_info "Clonage du repo SAM3..."
    git clone https://github.com/facebookresearch/sam3.git --quiet
fi

cd sam3
log_info "Installation du package SAM3..."
pip install -e . --quiet
cd ..
log_success "SAM3 installé"

# Installer Depth Anything 3
log_info "Installation de Depth Anything 3 depuis GitHub..."
if [ -d "Depth-Anything-3" ]; then
    log_warning "Depth Anything 3 déjà cloné, mise à jour..."
    cd Depth-Anything-3
    git pull --quiet
    cd ..
else
    log_info "Clonage du repo Depth Anything 3..."
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git --quiet
fi

cd Depth-Anything-3
log_info "Installation de xformers..."
pip install xformers --quiet || log_warning "xformers non installé (optionnel)"
log_info "Installation du package Depth Anything 3..."
pip install -e . --quiet
cd ..
log_success "Depth Anything 3 installé"

# Retour au dossier d'origine
cd "$SCRIPT_DIR"

# Vérifications finales
echo ""
log_info "Vérification des installations..."

# Vérifier les imports
python3 << EOF
import sys
errors = []

try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
except ImportError as e:
    errors.append(f"  ✗ PyTorch: {e}")

try:
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__}")
except ImportError as e:
    errors.append(f"  ✗ OpenCV: {e}")

try:
    from PySide6 import QtCore
    print(f"  ✓ PySide6 {QtCore.__version__}")
except ImportError as e:
    errors.append(f"  ✗ PySide6: {e}")

try:
    import PIL
    print(f"  ✓ Pillow {PIL.__version__}")
except ImportError as e:
    errors.append(f"  ✗ Pillow: {e}")

try:
    import decord
    print(f"  ✓ Decord {decord.__version__}")
except ImportError as e:
    errors.append(f"  ✗ Decord (requis par SAM3): {e}")

try:
    from sam3.model_builder import build_sam3_image_model
    print(f"  ✓ SAM3")
except ImportError as e:
    errors.append(f"  ✗ SAM3: {e}")

try:
    from depth_anything_3.api import DepthAnything3
    print(f"  ✓ Depth Anything 3")
except ImportError as e:
    errors.append(f"  ✗ Depth Anything 3: {e}")

if errors:
    print("\nErreurs détectées:")
    for err in errors:
        print(err)
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    log_success "Toutes les vérifications sont passées !"
else
    log_error "Certaines vérifications ont échoué"
    exit 1
fi

# Créer un script d'activation rapide
ACTIVATE_SCRIPT="$HOME/Documents/activate_venv_sam.sh"
cat > "$ACTIVATE_SCRIPT" << 'ACTIVATE_EOF'
#!/bin/bash
# Script d'activation rapide pour venv_sam

source "$HOME/Documents/venv_sam/bin/activate"

echo "✓ Environnement virtuel venv_sam activé"
echo ""
echo "Pour lancer l'application:"
echo "  cd /chemin/vers/sam4-main"
echo "  python run.py"
echo ""
echo "Pour désactiver l'environnement:"
echo "  deactivate"
ACTIVATE_EOF

chmod +x "$ACTIVATE_SCRIPT"
log_success "Script d'activation créé: $ACTIVATE_SCRIPT"

# Créer un alias dans .bashrc (optionnel)
if [ -f "$HOME/.bashrc" ]; then
    if ! grep -q "alias venv_sam" "$HOME/.bashrc"; then
        log_info "Ajout d'un alias 'venv_sam' dans .bashrc..."
        echo "" >> "$HOME/.bashrc"
        echo "# Alias pour activer venv_sam" >> "$HOME/.bashrc"
        echo "alias venv_sam='source $HOME/Documents/venv_sam/bin/activate'" >> "$HOME/.bashrc"
        log_success "Alias ajouté. Rechargez avec: source ~/.bashrc"
    fi
fi

# Résumé final
echo ""
echo -e "${GREEN}========================================"
echo "Installation terminée avec succès !"
echo -e "========================================${NC}"
echo ""
echo "📁 Environnement virtuel: $VENV_PATH"
echo "📦 Modèles externes: $MODELS_DIR"
echo ""
echo "Pour activer l'environnement:"
echo "  source $HOME/Documents/venv_sam/bin/activate"
echo "  # ou"
echo "  source $ACTIVATE_SCRIPT"
echo "  # ou (après source ~/.bashrc)"
echo "  venv_sam"
echo ""
echo "Pour lancer l'application SAM3 Roto Ultimate:"
echo "  cd $(pwd)"
echo "  source $HOME/Documents/venv_sam/bin/activate"
echo "  python run.py"
echo ""
echo "Prochaines étapes:"
echo "  1. Activer le venv"
echo "  2. Lancer python run.py"
echo "  3. Charger les modèles SAM3 et DA3 dans l'interface"
echo ""
echo -e "${BLUE}Bonne utilisation ! 🎬✨${NC}"
