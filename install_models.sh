#!/bin/bash
# Script d'installation pour SAM3 et Depth Anything 3
# Usage: bash install_models.sh

set -e

echo "========================================"
echo "Installation SAM3 + Depth Anything 3"
echo "========================================"

# Vérifier que python3 est disponible
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé"
    exit 1
fi

# Vérifier que git est disponible
if ! command -v git &> /dev/null; then
    echo "❌ Git n'est pas installé"
    exit 1
fi

# Créer un répertoire pour les repos
mkdir -p .external_models
cd .external_models

# Installer SAM3
echo ""
echo "📦 Installation de SAM3..."
if [ -d "sam3" ]; then
    echo "⚠️  Le dossier sam3 existe déjà, mise à jour..."
    cd sam3
    git pull
    cd ..
else
    git clone https://github.com/facebookresearch/sam3.git
fi

cd sam3
echo "Installation des dépendances SAM3..."
pip install -e .
cd ..

# Installer Depth Anything 3
echo ""
echo "📦 Installation de Depth Anything 3..."
if [ -d "Depth-Anything-3" ]; then
    echo "⚠️  Le dossier Depth-Anything-3 existe déjà, mise à jour..."
    cd Depth-Anything-3
    git pull
    cd ..
else
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
fi

cd Depth-Anything-3
echo "Installation des dépendances Depth Anything 3..."
pip install xformers
pip install -e .
cd ..

cd ..

echo ""
echo "✅ Installation terminée!"
echo ""
echo "Pour télécharger les checkpoints des modèles:"
echo ""
echo "SAM3:"
echo "  - Les checkpoints seront téléchargés automatiquement depuis HuggingFace"
echo "  - Modèles disponibles: facebook/sam3-hiera-large, facebook/sam3-hiera-base"
echo "  - Authentification HuggingFace peut être requise: huggingface-cli login"
echo ""
echo "Depth Anything 3:"
echo "  - Modèles disponibles: depth-anything/DA3-BASE, depth-anything/DA3-LARGE"
echo "  - depth-anything/DA3NESTED-GIANT-LARGE (meilleur qualité)"
echo "  - Téléchargement automatique depuis HuggingFace"
echo ""
echo "Lancez l'application avec: python run.py"
