# Installation Rapide - SAM3 Roto Ultimate

Ce guide permet d'installer rapidement l'application avec toutes ses dépendances dans un environnement virtuel propre.

---

## 🚀 Installation Automatique (Recommandée)

### Étape 1 : Télécharger le projet

```bash
cd ~/Downloads
git clone https://github.com/nebegreg/sam4.git sam4-main
cd sam4-main
git checkout claude/analyze-app-archive-01Qme7Y6vtqGVGRXBW2BwkKF
```

### Étape 2 : Lancer le script d'installation

```bash
bash install_venv_complete.sh
```

Le script va :
- ✅ Créer un venv dans `~/Documents/venv_sam`
- ✅ Installer PyTorch (avec CUDA si GPU disponible)
- ✅ Installer toutes les dépendances Python
- ✅ Cloner et installer SAM3 depuis GitHub
- ✅ Cloner et installer Depth Anything 3 depuis GitHub
- ✅ Vérifier que tout fonctionne
- ✅ Créer un script d'activation rapide

**Durée estimée : 5-10 minutes** (selon votre connexion Internet)

### Étape 3 : Activer l'environnement et lancer l'app

```bash
# Activer le venv
source ~/Documents/venv_sam/bin/activate

# Lancer l'application
python run.py
```

---

## 🔧 Installation Manuelle

Si vous préférez installer manuellement :

### 1. Créer l'environnement virtuel

```bash
python3 -m venv ~/Documents/venv_sam
source ~/Documents/venv_sam/bin/activate
```

### 2. Installer les dépendances de base

```bash
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

### 3. Installer SAM3

```bash
cd ~/Documents/venv_sam
mkdir -p .external_models
cd .external_models

git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..
```

### 4. Installer Depth Anything 3

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
pip install xformers
pip install -e .
cd ..
```

### 5. Retour au projet et lancement

```bash
cd ~/Downloads/sam4-main
source ~/Documents/venv_sam/bin/activate
python run.py
```

---

## 📦 Utilisation Quotidienne

### Activer l'environnement (3 méthodes)

**Méthode 1 : Commande directe**
```bash
source ~/Documents/venv_sam/bin/activate
```

**Méthode 2 : Script d'activation**
```bash
source ~/Documents/activate_venv_sam.sh
```

**Méthode 3 : Alias (après `source ~/.bashrc`)**
```bash
venv_sam
```

### Lancer l'application

```bash
cd ~/Downloads/sam4-main
source ~/Documents/venv_sam/bin/activate
python run.py
```

### Désactiver l'environnement

```bash
deactivate
```

---

## 🐛 Dépannage

### Erreur : "Python 3.12+ requis"

**Solution :**
```bash
# Sur CentOS/RHEL
sudo yum install python3.12

# Ou installer depuis les sources
```

### Erreur : "Git n'est pas installé"

**Solution :**
```bash
sudo yum install git
```

### Erreur : "CUDA out of memory"

**Solution :** Utiliser des modèles plus petits :
- SAM3 : `facebook/sam3-hiera-base` au lieu de `large`
- DA3 : `depth-anything/DA3-BASE` au lieu de `LARGE`

### Erreur : "Cannot import sam3"

**Solution :**
```bash
source ~/Documents/venv_sam/bin/activate
cd ~/Documents/venv_sam/.external_models/sam3
pip install -e .
```

### Erreur Qt threading (QThread::wait, QObject::setParent)

**Solution :** Vérifier que vous avez la dernière version du code :
```bash
cd ~/Downloads/sam4-main
git pull origin claude/analyze-app-archive-01Qme7Y6vtqGVGRXBW2BwkKF
```

---

## 📊 Vérifier l'Installation

```bash
source ~/Documents/venv_sam/bin/activate

# Vérifier les packages installés
pip list | grep -E "torch|opencv|PySide6|sam3|depth-anything"

# Tester les imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "from PySide6 import QtCore; print('PySide6:', QtCore.__version__)"
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3: OK')"
python -c "from depth_anything_3.api import DepthAnything3; print('DA3: OK')"
```

Tous les tests doivent passer sans erreur.

---

## 💾 Espace Disque Requis

- **Environnement virtuel** : ~3 GB
- **SAM3 repo** : ~500 MB
- **DA3 repo** : ~1 GB
- **Checkpoints SAM3** (téléchargés au premier usage) : ~2-3 GB
- **Checkpoints DA3** (téléchargés au premier usage) : ~1-5 GB

**Total : ~10-15 GB**

---

## 🔄 Mise à Jour

Pour mettre à jour l'application :

```bash
cd ~/Downloads/sam4-main
git pull origin claude/analyze-app-archive-01Qme7Y6vtqGVGRXBW2BwkKF

# Si nécessaire, mettre à jour les dépendances
source ~/Documents/venv_sam/bin/activate
pip install -r requirements.txt --upgrade
```

Pour mettre à jour SAM3 ou DA3 :

```bash
cd ~/Documents/venv_sam/.external_models/sam3
git pull
pip install -e .

cd ~/Documents/venv_sam/.external_models/Depth-Anything-3
git pull
pip install -e .
```

---

## 📞 Support

Si vous rencontrez des problèmes :

1. Vérifier que le venv est activé : `which python` doit afficher `~/Documents/venv_sam/bin/python`
2. Vérifier les logs d'erreur dans le terminal
3. Consulter les guides :
   - `CHECKPOINTS_GUIDE.md` - Pour les modèles
   - `ADVANCED_MATTING_GUIDE.md` - Pour le matting avancé
   - `README.md` - Documentation générale

---

## ✨ Première Utilisation

Après installation :

1. **Lancer l'app** : `python run.py`
2. **Charger SAM3** :
   - Entrer : `facebook/sam3-hiera-large` (ou chemin local)
   - Cliquer "⚙️ Charger SAM3"
   - Attendre "✅ SAM3 chargé."
3. **Importer une image/vidéo** :
   - "📼 Import vidéo" ou "🖼️ Import suite"
4. **Segmenter** :
   - Choisir mode PCS ou PVS
   - Ajouter prompts texte ou points
   - "▶ Segment frame"
5. **Raffiner** :
   - Onglet "Matte" → Choisir un preset
   - Ajuster les paramètres
   - Preview le résultat
6. **Exporter** :
   - Onglet "Export"
   - Choisir format (PNG, ProRes4444)

**Consultez le README.md pour le workflow complet !**

---

**Installation réussie ! 🎉**
