# 🛠️ Guide de Résolution - Segfault et Erreurs Qt

**Date**: 2025-11-28
**Problème**: Segmentation fault et erreurs Qt threading

---

## 🔍 Symptômes

```
QObject::setParent: Cannot set parent, new parent is in a different thread
Segmentation fault (core dumped)
```

```
ModuleNotFoundError: No module named 'transformers'
ModuleNotFoundError: No module named 'sam3'
```

---

## ✅ Solution 1: Installer les Dépendances

### Installation Rapide

```bash
# Activer votre environnement virtuel
source venv_sam3_ultimate/bin/activate
# ou votre environnement

# Installer transformers et dépendances
pip install transformers einops timm

# Installer SAM3 depuis GitHub
cd /tmp
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### Vérification

```bash
# Tester les imports
python3 -c "import transformers; print('transformers OK')"
python3 -c "import sam3; print('sam3 OK')"
python3 -c "import torch; print('torch OK')"
```

---

## ✅ Solution 2: Correction Qt Threading (DÉJÀ APPLIQUÉE)

Le problème de threading Qt a été corrigé dans `sam3roto/app.py`:

### Avant (Causait des segfaults):
```python
def _run_thread(self, fn, tag: str):
    th = QtCore.QThread()  # Variable locale
    wk = Worker(fn)         # Variable locale
    # ... setup ...
    th.start()              # Risque de garbage collection!
```

### Après (Fixé):
```python
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # ... init ...
        self._active_threads: List[Tuple[QtCore.QThread, Worker]] = []

    def _run_thread(self, fn, tag: str):
        th = QtCore.QThread()
        wk = Worker(fn)

        # Store to prevent garbage collection
        self._active_threads.append((th, wk))

        def cleanup():
            # Remove when done
            try:
                self._active_threads.remove((th, wk))
            except ValueError:
                pass
            wk.deleteLater()
            th.deleteLater()

        th.finished.connect(cleanup)
        th.start()
```

**Cause**: Python garbage collectait les objets `QThread` et `Worker` avant que Qt ne les utilise, causant des segfaults.

**Solution**: Stocker les threads actifs dans une liste d'instance (`self._active_threads`) empêche le garbage collector de les supprimer prématurément.

---

## ✅ Solution 3: Utiliser le Script de Lancement

Un script automatique vérifie les dépendances avant de lancer:

```bash
./launch_sam3roto.sh
```

Le script:
- ✅ Vérifie PyTorch, Transformers, PySide6
- ✅ Installe les dépendances manquantes
- ✅ Affiche des messages clairs
- ✅ Lance l'application

---

## 📋 Checklist de Dépannage

1. **Vérifier l'environnement virtuel**
   ```bash
   which python3
   # Doit pointer vers votre venv
   ```

2. **Installer toutes les dépendances**
   ```bash
   pip install -r requirements.txt
   pip install transformers einops timm
   ```

3. **Vérifier SAM3**
   ```bash
   python3 -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 OK')"
   ```

4. **Tester sans GUI**
   ```bash
   python3 -c "
   from sam3roto.backend.sam3_backend import SAM3Backend
   backend = SAM3Backend()
   print('Backend OK')
   "
   ```

5. **Lancer avec le script**
   ```bash
   ./launch_sam3roto.sh
   ```

---

## 🔧 Dépendances Requises

### Critiques
- ✅ Python 3.10+
- ✅ PyTorch 2.0+
- ✅ PySide6
- ✅ NumPy
- ✅ Pillow

### Pour SAM3
- ✅ transformers (ou)
- ✅ sam3 (GitHub repo)
- ✅ einops
- ✅ timm

### Installation Complète
```bash
# Base
pip install torch torchvision PySide6 numpy pillow opencv-python tqdm imageio psutil

# SAM3 via transformers
pip install transformers einops timm

# SAM3 via GitHub
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# Dépendances video
pip install decord pycocotools
```

---

## 🐛 Erreurs Courantes

### 1. "No module named 'transformers'"
**Solution**: `pip install transformers einops timm`

### 2. "No module named 'sam3'"
**Solution**: Installer depuis GitHub (voir ci-dessus) OU utiliser transformers

### 3. "QObject::setParent: Cannot set parent..."
**Solution**: Déjà corrigé dans `sam3roto/app.py` (commit actuel)

### 4. Segmentation fault au démarrage
**Causes possibles**:
- Threads Qt mal gérés ✅ (fixé)
- Dépendances incompatibles
- PyTorch/CUDA mismatch

**Solutions**:
1. Vérifier que la correction Qt est appliquée (git pull)
2. Réinstaller PyTorch compatible avec votre système
3. Vérifier les logs d'erreur avant le segfault

### 5. "CUDA error: no kernel image"
**Solution**: PyTorch CPU vs GPU mismatch
```bash
# Pour CPU uniquement
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Pour CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 📚 Fichiers Modifiés (Correction)

### sam3roto/app.py
- **Ligne 108**: Ajout de `self._active_threads: List[Tuple[QtCore.QThread, Worker]] = []`
- **Ligne 1060**: Store threads: `self._active_threads.append((th, wk))`
- **Lignes 1094-1098**: Cleanup avec remove: `self._active_threads.remove((th, wk))`

### Nouveau Fichier
- **launch_sam3roto.sh**: Script de lancement avec vérification automatique

---

## 🎯 Test de Validation

Pour vérifier que tout fonctionne:

```bash
# 1. Lancer le script de test
./launch_sam3roto.sh

# 2. Dans l'interface:
#    - Charger SAM3 (bouton "⚙️ Charger SAM3")
#    - Vérifier qu'il n'y a pas de segfault
#    - Vérifier le message "✅ SAM3 chargé."

# 3. Tester une segmentation simple
#    - Importer une image
#    - Mode: "Concept (PCS) image"
#    - Text: "person"
#    - Cliquer "▶ Segment frame"
```

---

## 💡 Prévention Future

### Bonnes Pratiques Qt/Python

1. **Toujours stocker les QObjects actifs**
   ```python
   # BAD
   thread = QThread()
   thread.start()  # Peut être GC!

   # GOOD
   self.threads.append(thread)
   thread.start()
   ```

2. **Utiliser parent=None pour Worker**
   ```python
   # QObject sans parent pour éviter conflits de threads
   worker = Worker(func)
   worker.moveToThread(thread)  # OK avec parent=None
   ```

3. **Toujours cleanup avec deleteLater()**
   ```python
   def cleanup():
       worker.deleteLater()
       thread.deleteLater()
   thread.finished.connect(cleanup)
   ```

---

## 🆘 Support

Si le problème persiste:

1. **Vérifier les logs**: `python3 run.py 2>&1 | tee app.log`
2. **Tester minimal**: Utiliser `test_sam3_loading.py`
3. **Environnement propre**: Recréer le venv
4. **Vérifier versions**:
   ```bash
   python3 --version
   python3 -c "import torch; print(torch.__version__)"
   python3 -c "import PySide6; print(PySide6.__version__)"
   ```

---

**Dernière mise à jour**: 2025-11-28
**Version**: 1.0
**Correction appliquée**: sam3roto/app.py (commit actuel)
