#!/usr/bin/env python3
"""
Script de diagnostic complet pour SAM3 Roto Ultimate
Teste toutes les dépendances et fonctionnalités
"""

import sys
from pathlib import Path

# Couleurs
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

# Test 1: Dépendances Python
print_header("Test 1: Dépendances Python")

deps_ok = True
required = {
    'numpy': 'NumPy',
    'torch': 'PyTorch',
    'PIL': 'Pillow',
    'PySide6': 'PySide6',
    'cv2': 'OpenCV',
}

for module, name in required.items():
    try:
        __import__(module)
        version = "OK"
        if module == 'torch':
            import torch
            version = torch.__version__
        elif module == 'numpy':
            import numpy
            version = numpy.__version__
        print_success(f"{name}: {version}")
    except ImportError as e:
        print_error(f"{name}: Non installé - {e}")
        deps_ok = False

if not deps_ok:
    print_error("\nDépendances manquantes! Installez:")
    print("  pip install torch torchvision numpy pillow PySide6 opencv-python")
    sys.exit(1)

# Test 2: Import du backend
print_header("Test 2: Import du backend SAM3")

sys.path.insert(0, str(Path(__file__).parent))

try:
    from sam3roto.backend.sam3_backend import SAM3Backend
    print_success("Import SAM3Backend: OK")
except Exception as e:
    print_error(f"Import SAM3Backend: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Création du backend
print_header("Test 3: Création du backend")

try:
    backend = SAM3Backend(enable_optimizations=False)
    print_success("Backend créé")
    print(f"  Device: {backend.device}")
    print(f"  Dtype: {backend.dtype}")
except Exception as e:
    print_error(f"Erreur création backend: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Vérifier SAM3 disponibilité
print_header("Test 4: Disponibilité SAM3")

transformers_ok = False
sam3_repo_ok = False

try:
    from transformers import Sam3Model, Sam3Processor
    print_success("Transformers SAM3: Disponible")
    transformers_ok = True
except ImportError as e:
    print_warning(f"Transformers SAM3: Non disponible - {e}")

try:
    from sam3.model_builder import build_sam3_image_model
    print_success("SAM3 GitHub repo: Disponible")
    sam3_repo_ok = True
except ImportError as e:
    print_warning(f"SAM3 GitHub repo: Non disponible - {e}")

if not transformers_ok and not sam3_repo_ok:
    print_error("\nAucune méthode de chargement SAM3 disponible!")
    print("\nSOLUTIONS:")
    print("1. Transformers: pip install transformers einops timm")
    print("2. GitHub repo:")
    print("   git clone https://github.com/facebookresearch/sam3.git")
    print("   cd sam3 && pip install -e .")
    sys.exit(1)

# Test 5: Chargement du modèle (optionnel - peut être long)
print_header("Test 5: Chargement du modèle (optionnel)")

print("Voulez-vous tester le chargement du modèle? (y/n)")
print("⚠ Cela peut prendre plusieurs minutes et télécharger ~2-4GB")
choice = input("> ").strip().lower()

if choice == 'y':
    try:
        print("\nChargement SAM3...")
        if transformers_ok:
            backend.load("facebook/sam3-hiera-tiny")  # Plus petit pour test
        elif sam3_repo_ok:
            backend.load("facebook/sam3")

        print_success("Modèle chargé avec succès!")

        # Test 6: Segmentation simple
        print_header("Test 6: Test de segmentation")

        try:
            from PIL import Image
            import numpy as np

            # Créer une image de test
            test_img = Image.new('RGB', (512, 512), color='white')

            # Tester segmentation
            print("Test segmentation avec prompt 'person'...")
            masks = backend.segment_concept_image(test_img, "person")

            print_success(f"Segmentation OK - {len(masks)} objets trouvés")

        except Exception as e:
            print_error(f"Erreur segmentation: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print_error(f"Erreur chargement: {e}")
        import traceback
        traceback.print_exc()
else:
    print_warning("Test de chargement ignoré")

# Test 7: Import de l'application
print_header("Test 7: Import de l'application")

try:
    from sam3roto.app import MainWindow
    print_success("Import MainWindow: OK")
except Exception as e:
    print_error(f"Import MainWindow: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Qt Application
print_header("Test 8: Qt Application (basique)")

try:
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    print_success("Qt Application: OK")
except Exception as e:
    print_error(f"Qt Application: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Résumé
print_header("RÉSUMÉ DU DIAGNOSTIC")

print("Statut des composants:")
print(f"  {'✓' if deps_ok else '✗'} Dépendances Python")
print(f"  {'✓' if transformers_ok or sam3_repo_ok else '✗'} SAM3 disponible")
print(f"    {'✓' if transformers_ok else '✗'} Via Transformers")
print(f"    {'✓' if sam3_repo_ok else '✗'} Via GitHub repo")

print("\nRecommandations:")
if not transformers_ok and not sam3_repo_ok:
    print("  ❗ Installer SAM3 (transformers ou GitHub repo)")
elif transformers_ok:
    print("  ✓ Tout semble OK! Utilisez: facebook/sam3-hiera-large")
elif sam3_repo_ok:
    print("  ✓ Tout semble OK! Utilisez: facebook/sam3")

print("\nPour lancer l'application:")
print("  python3 run.py")
print("  # ou")
print("  ./launch_sam3roto.sh")

print(f"\n{Colors.BOLD}Diagnostic terminé!{Colors.RESET}\n")
