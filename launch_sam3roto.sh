#!/bin/bash
# Script de lancement pour SAM3 Roto Ultimate
# Vérifie les dépendances et lance l'application

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo "SAM3 Roto Ultimate - Launcher"
echo -e "========================================${NC}"

# Fonction pour vérifier un module Python
check_module() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# Vérifier les dépendances critiques
echo -e "\n${BLUE}[INFO]${NC} Vérification des dépendances..."

missing=0

# PyTorch
if check_module "torch"; then
    version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo -e "${GREEN}✓${NC} PyTorch: $version"
else
    echo -e "${RED}✗${NC} PyTorch: Non installé"
    missing=1
fi

# Transformers
if check_module "transformers"; then
    version=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null)
    echo -e "${GREEN}✓${NC} Transformers: $version"
else
    echo -e "${YELLOW}⚠${NC} Transformers: Non installé (installé)"
    pip install transformers einops timm --quiet
    echo -e "${GREEN}✓${NC} Transformers: Installé"
fi

# PySide6
if check_module "PySide6"; then
    echo -e "${GREEN}✓${NC} PySide6: OK"
else
    echo -e "${RED}✗${NC} PySide6: Non installé"
    missing=1
fi

# SAM3 (optionnel - sera géré par le fallback)
if check_module "sam3"; then
    echo -e "${GREEN}✓${NC} SAM3 (GitHub repo): OK"
elif check_module "transformers"; then
    echo -e "${YELLOW}⚠${NC} SAM3 (GitHub repo): Non installé, utilisera transformers si disponible"
else
    echo -e "${YELLOW}⚠${NC} SAM3: Installation recommandée"
fi

# Si des dépendances manquent
if [ $missing -eq 1 ]; then
    echo -e "\n${RED}[ERROR]${NC} Dépendances manquantes!"
    echo -e "${YELLOW}Solution:${NC}"
    echo "  pip install torch torchvision PySide6 numpy pillow opencv-python"
    echo "  pip install transformers einops timm"
    exit 1
fi

echo -e "\n${GREEN}✓${NC} Toutes les dépendances critiques sont installées"

# Lancer l'application
echo -e "\n${BLUE}[INFO]${NC} Lancement de SAM3 Roto..."
cd "$(dirname "$0")"
python3 run.py

exit 0
