#!/usr/bin/env python3
"""
Test complet de l'installation SAM3 + Depth Anything 3
Date: 2025-12-03
Usage: python3 test_installation.py
"""

import sys

def test_imports():
    """Test tous les imports requis"""
    print("=== Test des Imports ===\n")

    tests = [
        ("PyTorch", "import torch; print(f'  Version: {torch.__version__}'); assert torch.cuda.is_available(), 'CUDA non disponible'"),
        ("CUDA", "import torch; print(f'  CUDA version: {torch.version.cuda}')"),
        ("torchvision", "import torchvision; print(f'  Version: {torchvision.__version__}')"),
        ("NumPy", "import numpy; print(f'  Version: {numpy.__version__}')"),
        ("Pillow", "import PIL; print(f'  Version: {PIL.__version__}')"),
        ("OpenCV", "import cv2; print(f'  Version: {cv2.__version__}')"),
        ("PySide6", "import PySide6; print('  Version: OK')"),
        ("xformers", "import xformers; print(f'  Version: {xformers.__version__}')"),
        ("transformers", "import transformers; print(f'  Version: {transformers.__version__}')"),
        ("einops", "import einops; print('  Version: OK')"),
        ("timm", "import timm; print(f'  Version: {timm.__version__}')"),
        ("scipy", "import scipy; print(f'  Version: {scipy.__version__}')"),
        ("matplotlib", "import matplotlib; print(f'  Version: {matplotlib.__version__}')"),
        ("scikit-learn", "import sklearn; print(f'  Version: {sklearn.__version__}')"),
    ]

    passed = 0
    failed = 0

    for name, code in tests:
        try:
            print(f"[TEST] {name}...", end=" ")
            exec(code)
            print("✓")
            passed += 1
        except Exception as e:
            print(f"✗ ÉCHEC: {e}")
            failed += 1

    print(f"\n✓ Réussis: {passed}/{len(tests)}")
    if failed > 0:
        print(f"✗ Échecs: {failed}/{len(tests)}")

    return failed == 0

def test_sam3():
    """Test SAM3"""
    print("\n=== Test SAM3 ===\n")

    transformers_ok = False
    github_ok = False

    # Test 1: Transformers API
    try:
        print("[TEST] Import transformers SAM3...", end=" ")
        from transformers import Sam3Model, Sam3Processor
        print("✓")
        print("  Méthode: transformers API")
        transformers_ok = True
    except ImportError as e:
        print("✗ (Non disponible)")
        print(f"  Raison: {e}")
        print("  Note: SAM3 pas encore dans release stable")

    # Test 2: GitHub Repo
    try:
        print("[TEST] Import sam3 GitHub...", end=" ")
        from sam3.model_builder import build_sam3_image_model
        print("✓")
        print("  Méthode: GitHub repo")
        github_ok = True
    except ImportError as e:
        print(f"✗ ÉCHEC: {e}")
        print("  Solution: cd /tmp && git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .")

    if transformers_ok or github_ok:
        print("\n✓ SAM3: Au moins une méthode disponible")
        return True
    else:
        print("\n✗ SAM3: Aucune méthode disponible")
        return False

def test_depth_anything():
    """Test Depth Anything 3"""
    print("\n=== Test Depth Anything 3 ===\n")

    try:
        print("[TEST] Import Depth Anything 3...", end=" ")
        from depth_anything_3.api import DepthAnything3
        print("✓")
        print("  Module: depth_anything_3.api")
        return True
    except ImportError as e:
        print(f"✗ ÉCHEC: {e}")
        print("  Solution: cd /tmp && git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git && cd Depth-Anything-3 && pip install -e .")
        return False

def test_cuda():
    """Test CUDA"""
    print("\n=== Test CUDA ===\n")

    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  VRAM totale: {props.total_memory / 1024**3:.2f} GB")
            print(f"  VRAM libre: {(props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3:.2f} GB")
            print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Multi-processor count: {props.multi_processor_count}")

        return True
    else:
        print("⚠ ATTENTION: CUDA n'est pas disponible!")
        print("\nSolutions possibles:")
        print("  1. Vérifier que le GPU est détecté: nvidia-smi")
        print("  2. Réinstaller PyTorch avec CUDA:")
        print("     pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \\")
        print("       --index-url https://download.pytorch.org/whl/cu128")
        return False

def test_environment():
    """Test l'environnement virtuel"""
    print("\n=== Test Environnement ===\n")

    import sys
    import os

    python_path = sys.executable
    venv_activated = "venv_sam3_ultimate" in python_path or "sam3_env" in python_path

    print(f"Python executable: {python_path}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Virtual env activé: {'✓ OUI' if venv_activated else '✗ NON'}")

    if "VIRTUAL_ENV" in os.environ:
        print(f"VIRTUAL_ENV: {os.environ['VIRTUAL_ENV']}")
    elif "CONDA_DEFAULT_ENV" in os.environ:
        print(f"CONDA_DEFAULT_ENV: {os.environ['CONDA_DEFAULT_ENV']}")

    if not venv_activated:
        print("\n⚠ ATTENTION: Environnement virtuel pas activé!")
        print("Activer avec:")
        print("  source ~/venv_sam3_ultimate/bin/activate")
        print("  OU")
        print("  conda activate sam3_env")

    return venv_activated

def test_huggingface_auth():
    """Test authentification HuggingFace"""
    print("\n=== Test HuggingFace Authentication ===\n")

    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()

        if token:
            print("✓ Token HuggingFace trouvé")
            print(f"  Token: {token[:10]}...{token[-10:]}")
            return True
        else:
            print("✗ Pas de token HuggingFace")
            print("\nAuthentification requise pour SAM3:")
            print("  1. Créer token: https://huggingface.co/settings/tokens")
            print("  2. Demander accès: https://huggingface.co/facebook/sam3")
            print("  3. Se connecter: huggingface-cli login")
            return False
    except ImportError:
        print("✗ huggingface_hub pas installé")
        print("  Installation: pip install huggingface-hub")
        return False

def main():
    print("=" * 60)
    print("Test d'Installation SAM3 + Depth Anything 3")
    print("=" * 60)

    results = []

    results.append(("Environnement virtuel", test_environment()))
    results.append(("Imports de base", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("SAM3", test_sam3()))
    results.append(("Depth Anything 3", test_depth_anything()))
    results.append(("HuggingFace Auth", test_huggingface_auth()))

    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)

    for name, passed in results:
        status = "✓ OK" if passed else "✗ ÉCHEC"
        print(f"{name:.<40} {status}")

    all_passed = all(passed for _, passed in results)
    critical_passed = results[0][1] and results[1][1] and results[2][1]  # Env, Imports, CUDA

    print("\n" + "=" * 60)

    if all_passed:
        print("🎉 Installation complète et fonctionnelle!")
        print("\nVous pouvez lancer l'application:")
        print("  python3 run.py")
        return 0
    elif critical_passed:
        print("⚠ Installation fonctionnelle mais incomplète")
        print("\nL'application devrait fonctionner, mais:")
        print("  - Certaines fonctionnalités peuvent manquer")
        print("  - SAM3 nécessite authentification HuggingFace")
        print("\nVous pouvez essayer de lancer l'application:")
        print("  python3 run.py")
        return 0
    else:
        print("✗ Installation incomplète - Composants critiques manquants")
        print("\nCorrigez les erreurs ci-dessus avant de continuer.")
        print("Relancez le script d'installation:")
        print("  ./install_sam3_da3.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())
