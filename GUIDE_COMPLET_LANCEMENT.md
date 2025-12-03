# 🚀 Guide Complet de Lancement - SAM3 Roto Ultimate

**PROBLÈME IDENTIFIÉ**: Vous n'êtes pas dans l'environnement virtuel!

---

## ✅ Solution: Activer l'environnement virtuel

### Étape 1: Trouver votre environnement

Vous avez mentionné `(sam3)`, donc votre environnement est probablement à:
- `/home/reepost/Documents/venv_sam/` OU
- `/home/reepost/.virtualenvs/sam3/` OU
- `~/venv_sam3_ultimate/`

**Vérifiez quel chemin existe:**

```bash
ls -la ~/Documents/venv_sam/bin/activate 2>/dev/null && echo "Trouvé: ~/Documents/venv_sam"
ls -la ~/.virtualenvs/sam3/bin/activate 2>/dev/null && echo "Trouvé: ~/.virtualenvs/sam3"
ls -la ~/venv_sam3_ultimate/bin/activate 2>/dev/null && echo "Trouvé: ~/venv_sam3_ultimate"
```

### Étape 2: Activer l'environnement

Une fois que vous savez quel chemin est le bon:

```bash
# Si c'est venv_sam:
source ~/Documents/venv_sam/bin/activate

# Si c'est .virtualenvs/sam3:
source ~/.virtualenvs/sam3/bin/activate

# Si c'est venv_sam3_ultimate:
source ~/venv_sam3_ultimate/bin/activate
```

**Vérification**: Vous devez voir `(sam3)` ou `(venv_sam)` au début de votre prompt.

### Étape 3: Vérifier les dépendances

```bash
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import PySide6; print('PySide6: OK')"
```

**Si ça affiche les versions = OK!**
**Si erreurs = environnement incorrect**

### Étape 4: Lancer l'application

```bash
cd ~/Downloads/sam4-main   # Ou votre dossier
python3 run.py
```

---

## 🔧 Méthode Alternative: Script Automatique

J'ai créé un script qui active automatiquement l'environnement:

```bash
cd ~/Downloads/sam4-main
./launch_sam3roto.sh
```

---

## 📋 Diagnostic Complet

Pour identifier tous les problèmes:

```bash
# Activer l'environnement d'abord!
source ~/Documents/venv_sam/bin/activate

# Lancer le diagnostic
cd ~/Downloads/sam4-main
python3 diagnostic.py
```

---

## ⚠️ Erreurs Courantes

### Erreur 1: "No module named 'numpy'"

**Cause**: Environnement virtuel pas activé

**Solution**:
```bash
source ~/Documents/venv_sam/bin/activate
```

### Erreur 2: "ModuleNotFoundError: No module named 'transformers'"

**Cause**: Transformers pas installé dans ce venv

**Solution**:
```bash
# Dans le venv activé:
pip install transformers einops timm
```

### Erreur 3: "No module named 'sam3'"

**Cause**: SAM3 GitHub repo pas installé

**Solution**:
```bash
# Dans le venv activé:
cd /tmp
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### Erreur 4: Segmentation fault / QObject error

**Cause**: Thread garbage collection (DÉJÀ CORRIGÉ dans le code)

**Solution**: Git pull pour obtenir la dernière version
```bash
cd ~/Downloads/sam4-main
git pull origin claude/analyze-app-archive-01Qme7Y6vtqGVGRXBW2BwkKF
```

---

## 🎯 Procédure Complète (Étape par Étape)

### 1. Trouver et activer l'environnement

```bash
# Chercher l'environnement
find ~ -name "activate" -path "*/bin/activate" 2>/dev/null | grep -E "(sam|venv)"

# Exemple de sortie:
# /home/reepost/Documents/venv_sam/bin/activate

# Activer (ajuster le chemin):
source /home/reepost/Documents/venv_sam/bin/activate
```

### 2. Vérifier l'activation

```bash
which python3
# Doit afficher: /home/reepost/Documents/venv_sam/bin/python3
# PAS: /usr/bin/python3

echo $VIRTUAL_ENV
# Doit afficher: /home/reepost/Documents/venv_sam
```

### 3. Installer les dépendances manquantes

```bash
# Si transformers manque:
pip install transformers einops timm

# Si SAM3 manque:
cd /tmp
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### 4. Aller au dossier du projet

```bash
cd ~/Downloads/sam4-main
# ou
cd /home/reepost/Downloads/sam4-main\ (5)/sam4-main
```

### 5. Lancer l'application

```bash
python3 run.py
```

---

## 🧪 Test Minimal

Pour tester sans GUI:

```bash
# Dans le venv activé:
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

print("Test 1: Import NumPy...")
import numpy as np
print(f"✓ NumPy {np.__version__}")

print("\nTest 2: Import PyTorch...")
import torch
print(f"✓ PyTorch {torch.__version__}")

print("\nTest 3: Import SAM3Backend...")
from sam3roto.backend.sam3_backend import SAM3Backend
print("✓ SAM3Backend")

print("\nTest 4: Créer Backend...")
backend = SAM3Backend(enable_optimizations=False)
print(f"✓ Backend créé (device: {backend.device})")

print("\n✅ Tous les tests OK! L'application devrait fonctionner.")
EOF
```

Si ce script s'exécute sans erreur = tout est OK!

---

## 💡 Créer un Alias Permanent

Pour éviter de retaper la commande d'activation:

```bash
# Ajouter à ~/.bashrc:
echo 'alias sam3="source ~/Documents/venv_sam/bin/activate && cd ~/Downloads/sam4-main"' >> ~/.bashrc

# Recharger:
source ~/.bashrc

# Maintenant vous pouvez juste taper:
sam3
python3 run.py
```

---

## 🔍 Identifier l'Environnement Actuel

Si vous ne savez pas quel environnement vous utilisez:

```bash
# Méthode 1: Variable d'environnement
echo $VIRTUAL_ENV

# Méthode 2: Emplacement de Python
which python3

# Méthode 3: Packages installés
pip list | grep -E "(torch|transformers|sam)"
```

---

## 📝 Checklist Avant de Lancer

- [ ] Environnement virtuel activé (`(sam3)` dans le prompt)
- [ ] `which python3` pointe vers le venv (PAS `/usr/bin/python3`)
- [ ] `python3 -c "import numpy"` fonctionne
- [ ] `python3 -c "import torch"` fonctionne
- [ ] `python3 -c "import PySide6"` fonctionne
- [ ] Dans le bon dossier (`cd ~/Downloads/sam4-main`)
- [ ] Git à jour (`git pull`)

**Si toutes les cases sont cochées → `python3 run.py` devrait fonctionner!**

---

## 🆘 Si Ça Ne Marche Toujours Pas

1. **Copier TOUTE la sortie d'erreur**:
   ```bash
   python3 run.py 2>&1 | tee error.log
   ```

2. **Vérifier les versions**:
   ```bash
   python3 --version
   pip list | grep -E "(torch|numpy|PySide|transformers)"
   ```

3. **Réinstaller depuis zéro**:
   ```bash
   # Créer nouvel environnement
   python3 -m venv ~/venv_sam3_fresh
   source ~/venv_sam3_fresh/bin/activate

   # Installer tout
   pip install torch torchvision numpy pillow PySide6 opencv-python tqdm imageio psutil
   pip install transformers einops timm

   # Installer SAM3
   cd /tmp
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3
   pip install -e .

   # Lancer
   cd ~/Downloads/sam4-main
   python3 run.py
   ```

---

**Dernière mise à jour**: 2025-11-28
**Le problème le plus fréquent**: Environnement virtuel pas activé!
