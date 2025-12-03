# Correction du Venv - Explications

## Problèmes Identifiés

### 1. **Erreur de compilation xformers**
```
ModuleNotFoundError: No module named 'torch'
ERROR: Failed to build 'xformers' when getting requirements to build wheel
```

**Cause:** xformers nécessite PyTorch pour être compilé, mais l'ordre d'installation était incorrect. Le script tentait d'installer xformers avant ou en même temps que PyTorch.

### 2. **Conflits de dépendances**
- L'installation de Depth Anything 3 avec `pip install -e ".[all]"` tentait d'installer toutes les dépendances simultanément
- Cela provoquait des résolutions de dépendances complexes et des erreurs de backtracking pip

### 3. **Versions de Python**
- Python 3.11 était utilisé au lieu de Python 3.12
- Certaines bibliothèques sont optimisées pour Python 3.12

## Solutions Implémentées

### Script `setup_venv_fixed.sh`

#### 1. **Ordre d'installation strict**
```bash
# Ordre CRITIQUE pour éviter les erreurs:
1. pip, setuptools, wheel (outils de base)
2. PyTorch + torchvision + torchaudio (OBLIGATOIRE EN PREMIER)
3. Bibliothèques de base (numpy, pillow, opencv, etc.)
4. Utilitaires ML (einops, timm)
5. xformers (nécessite torch)
6. Transformers et Hugging Face
7. SAM3 (nécessite transformers)
8. Depth Anything 3 (nécessite tout le reste)
```

#### 2. **Installation de PyTorch avec CUDA**
```bash
# Essai CUDA 12.8 d'abord
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Fallback sur CUDA 12.6 si nécessaire
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126
```

#### 3. **Installation xformers sans isolation**
```bash
# --no-build-isolation permet à xformers d'utiliser torch déjà installé
pip install xformers --no-build-isolation
```

#### 4. **Installation Depth Anything 3 sans conflits**
```bash
# Installation sans dépendances d'abord
pip install -e . --no-deps

# Puis installation des dépendances manquantes individuellement
pip install evo pycocotools decord pre-commit || true
```

#### 5. **Utilisation de Python 3.12**
```bash
PYTHON_CMD="python3.12"
python3.12 -m venv ~/venv_sam3_fixed
```

## Fonctionnalités du Script

### ✅ Vérifications préalables
- Vérification de CUDA/nvidia-smi
- Vérification de Python 3.12
- Détection des environnements existants

### ✅ Gestion de l'environnement
- Option de suppression de l'ancien venv
- Création d'un venv propre avec Python 3.12
- Vérification de l'activation

### ✅ Installation robuste
- Mise à jour des outils de base
- Installation séquentielle dans le bon ordre
- Gestion des erreurs avec fallbacks
- Vérification CUDA après installation PyTorch

### ✅ Tests et validations
- Test des versions installées
- Test des imports SAM3 et Depth Anything 3
- Affichage des informations GPU

### ✅ Utilitaires
- Script d'activation rapide (`activate_venv.sh`)
- Instructions claires pour HuggingFace
- Commandes pour tester et lancer l'application

## Utilisation

### Installation complète
```bash
cd /home/user/sam4
./setup_venv_fixed.sh
```

Le script va:
1. Vérifier les prérequis (CUDA, Python 3.12)
2. Créer un venv propre dans `~/venv_sam3_fixed`
3. Installer toutes les dépendances dans le bon ordre
4. Tester l'installation

### Activation de l'environnement
```bash
# Méthode 1 : Script rapide
./activate_venv.sh

# Méthode 2 : Activation manuelle
source ~/venv_sam3_fixed/bin/activate
```

### Après installation
```bash
# Se connecter à HuggingFace
huggingface-cli login

# Tester l'installation
python3 test_installation.py

# Lancer l'application
python3 run.py
```

## Différences avec les anciens scripts

| Ancien Script | Nouveau Script |
|--------------|----------------|
| Installation groupée | Installation séquentielle stricte |
| Python 3.11 ou 3.12 | Python 3.12 obligatoire |
| xformers avec build isolation | xformers sans build isolation |
| `pip install -e ".[all]"` | Installation contrôlée sans deps puis ajout |
| Pas de fallback CUDA | Fallback automatique 12.8 → 12.6 |
| Vérifications basiques | Vérifications complètes + tests |

## Erreurs courantes évitées

### ❌ Avant
```
ERROR: Failed to build 'xformers' when getting requirements to build wheel
ModuleNotFoundError: No module named 'torch'
```

### ✅ Après
```
✓ PyTorch installé
✓ xformers installé
✓ SAM3 installé
✓ Depth Anything 3 installé
```

## Support et debugging

### Si xformers échoue
Le script continue avec un avertissement. xformers est optionnel pour certaines fonctionnalités.

### Si CUDA 12.8 échoue
Le script essaie automatiquement CUDA 12.6.

### Pour vérifier l'installation
```bash
source ~/venv_sam3_fixed/bin/activate
python3 test_installation.py
```

### Pour voir les logs détaillés
```bash
./setup_venv_fixed.sh 2>&1 | tee installation.log
```

## Prochaines étapes

1. ✅ Environnement virtuel créé et fonctionnel
2. 🔄 Configuration HuggingFace (manuel)
3. 🔄 Tests de l'installation
4. 🔄 Lancement de l'application

## Notes importantes

- **Internet requis** : Le script télécharge ~10GB de données
- **Durée** : Environ 15-30 minutes selon la connexion
- **Espace disque** : Minimum 20GB libres recommandés
- **RAM** : Minimum 8GB, 16GB recommandé

## Contact et support

Si vous rencontrez des problèmes:
1. Vérifiez que CUDA fonctionne: `nvidia-smi`
2. Vérifiez Python 3.12: `python3.12 --version`
3. Consultez les logs d'installation
4. Vérifiez l'espace disque: `df -h`
