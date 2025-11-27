#!/usr/bin/env python3
"""
Script de diagnostic pour tester le chargement de SAM3
"""

import sys
import traceback

print("=" * 70)
print("🔍 SAM3 Loading Diagnostic Tool")
print("=" * 70)
print()

# Test 1: Transformers availability
print("📋 Test 1: Vérification de transformers...")
try:
    import transformers
    print(f"   ✓ transformers version: {transformers.__version__}")

    # Check if Sam3Model exists
    try:
        from transformers import Sam3Model, Sam3Processor
        print("   ✓ Sam3Model trouvé dans transformers")
        transformers_sam3_available = True
    except ImportError:
        print("   ✗ Sam3Model non disponible dans transformers")
        print("   → SAM3 n'est pas encore dans la version stable de transformers")
        transformers_sam3_available = False

except ImportError:
    print("   ✗ transformers non installé")
    transformers_sam3_available = False

print()

# Test 2: Check for SAM2 (fallback)
print("📋 Test 2: Vérification de SAM2 (fallback)...")
try:
    from transformers import Sam2Model, Sam2Processor
    print("   ✓ Sam2Model disponible")
    print("   → Peut être utilisé en mode compatibilité limitée")
    sam2_available = True
except ImportError:
    print("   ✗ Sam2Model non disponible")
    sam2_available = False

print()

# Test 3: GitHub repo installation
print("📋 Test 3: Vérification du repo GitHub SAM3...")
try:
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
    from sam3.model.sam3_image_processor import Sam3Processor
    print("   ✓ Repo GitHub SAM3 installé et importable")
    github_sam3_available = True
except ImportError as e:
    print(f"   ✗ Repo GitHub SAM3 non installé: {e}")
    print("   → Installez avec:")
    print("      cd ~/Documents/venv_sam/.external_models")
    print("      git clone https://github.com/facebookresearch/sam3.git")
    print("      cd sam3")
    print("      pip install -e .")
    github_sam3_available = False

print()

# Test 4: Dependencies
print("📋 Test 4: Vérification des dépendances...")
deps_ok = True

try:
    import torch
    print(f"   ✓ torch: {torch.__version__}")
except ImportError:
    print("   ✗ torch non installé")
    deps_ok = False

try:
    import decord
    print(f"   ✓ decord: {decord.__version__}")
except ImportError:
    print("   ✗ decord non installé (pip install decord)")
    deps_ok = False

try:
    import pycocotools
    print("   ✓ pycocotools installé")
except ImportError:
    print("   ✗ pycocotools non installé (pip install pycocotools)")
    deps_ok = False

print()
print("=" * 70)
print("📊 RÉSUMÉ")
print("=" * 70)
print()

if transformers_sam3_available:
    print("✅ MÉTHODE 1 (Transformers): DISPONIBLE")
    print("   → Utilisez: 'facebook/sam3-hiera-large' dans l'interface")
elif sam2_available:
    print("⚠️  MÉTHODE 1 (SAM2): DISPONIBLE (fonctionnalités limitées)")
    print("   → Utilisez: 'facebook/sam2-hiera-large' dans l'interface")
else:
    print("❌ MÉTHODE 1 (Transformers): NON DISPONIBLE")

print()

if github_sam3_available:
    print("✅ MÉTHODE 2 (Repo GitHub): DISPONIBLE")
    print("   → Téléchargez un checkpoint et utilisez le chemin local")
else:
    print("❌ MÉTHODE 2 (Repo GitHub): NON DISPONIBLE")
    print("   → Exécutez: bash install_venv_complete.sh")

print()

if not (transformers_sam3_available or github_sam3_available):
    print("🔧 ACTIONS RECOMMANDÉES:")
    print()
    print("1. OPTION SIMPLE - Installer le repo GitHub:")
    print("   bash install_venv_complete.sh")
    print()
    print("2. OPTION ALTERNATIVE - SAM2 (limité):")
    print("   pip install --upgrade transformers")
    print("   # Puis utilisez 'facebook/sam2-hiera-large'")
    print()
else:
    print("✅ Au moins une méthode est disponible!")
    print("   Vous pouvez charger SAM3 dans l'application")

print()
print("=" * 70)
