# Guide de Débogage - SAM3 Roto

## 🐛 Problèmes Qt / Segfault

Si vous rencontrez des erreurs comme:
```
QObject::setParent: Cannot set parent, new parent is in a different thread
QBackingStore::endPaint() called with active painter
Segmentation fault (core dumped)
```

Ces erreurs indiquent des **problèmes de threading avec Qt/PySide6**.

## 📊 Lancer avec le logging activé

### Méthode recommandée

Utilisez le script de démarrage avec logging:

```bash
python3 run_with_logging.py
```

Ce script:
- ✅ Active le logging détaillé
- ✅ Capture les erreurs et exceptions
- ✅ Sauvegarde les logs dans `~/.sam3roto/logs/`
- ✅ Affiche le chemin du fichier de log

### Sortie attendue

```
================================================================================
SAM3 Roto Ultimate - Démarrage avec logging activé
================================================================================
✓ Logging configuré: /home/user/.sam3roto/logs/sam3roto_20251203_143052.log
[Main] SAM3Roto | INFO | ============================================================
[Main] SAM3Roto | INFO | SAM3 Roto Ultimate - Démarrage
[Main] SAM3Roto | INFO | ============================================================
[MainThread] Main | INFO | Import de PySide6...
[MainThread] Main | INFO | ✓ PySide6 importé
[MainThread] Main | INFO | Création QApplication...
[MainThread] Main | INFO | ✓ QApplication créée
[MainThread] Main | INFO | Import de l'application principale...
[MainThread] Main | INFO | ✓ Application importée
[MainThread] Main | INFO | Création de la fenêtre principale...
[MainThread] Main | INFO | ✓ Fenêtre principale créée
[MainThread] Main | INFO | Affichage de la fenêtre...
[MainThread] Main | INFO | ✓ Fenêtre affichée
[MainThread] Main | INFO | Démarrage de la boucle événementielle Qt...

✓ Application démarrée. Consultez les logs pour les détails.
```

## 📁 Fichiers de log

Les logs sont sauvegardés dans:
```
~/.sam3roto/logs/sam3roto_YYYYMMDD_HHMMSS.log
```

Format des logs:
```
2025-12-03 14:30:52 | MainThread      | SAM3Backend          | INFO     | track_concept_video: début (frames=60, texts=['person'])
2025-12-03 14:30:52 | MainThread      | SAM3Backend          | DEBUG    | track_concept_video: temp_dir=/tmp/sam3_video_abc123
2025-12-03 14:30:53 | MainThread      | SAM3Backend          | INFO     | [SAM3 Video] Saving 60 frames to temp dir...
2025-12-03 14:30:55 | MainThread      | SAM3Backend          | INFO     | [SAM3 Video] Frames saved
2025-12-03 14:30:55 | MainThread      | SAM3Backend          | INFO     | [SAM3 Video] Starting video session...
2025-12-03 14:30:55 | MainThread      | SAM3Backend          | DEBUG    | track_concept_video: calling handle_request(type=start_session)
```

## 🔍 Analyser un crash

### 1. Après un segfault

```bash
# Trouver le dernier log
ls -lt ~/.sam3roto/logs/ | head -1

# Lire le log
tail -100 ~/.sam3roto/logs/sam3roto_YYYYMMDD_HHMMSS.log
```

### 2. Chercher les erreurs

```bash
# Trouver les ERROR dans le log
grep "ERROR" ~/.sam3roto/logs/sam3roto_YYYYMMDD_HHMMSS.log

# Trouver les WARNING
grep "WARNING" ~/.sam3roto/logs/sam3roto_YYYYMMDD_HHMMSS.log

# Dernières lignes avant le crash
tail -50 ~/.sam3roto/logs/sam3roto_YYYYMMDD_HHMMSS.log
```

### 3. Identifier la dernière opération

Les logs montrent exactement ce qui se passait avant le crash:

```
2025-12-03 14:30:55 | MainThread      | SAM3Backend          | DEBUG    | track_concept_video: calling handle_request(type=start_session)
2025-12-03 14:30:56 | MainThread      | SAM3Backend          | DEBUG    | track_concept_video: handle_request returned: {'session_id': '12345'}
2025-12-03 14:30:56 | MainThread      | SAM3Backend          | INFO     | [SAM3 Video] Session started: 12345
2025-12-03 14:30:56 | MainThread      | SAM3Backend          | DEBUG    | track_concept_video: adding prompt 0/1: 'person'
[SEGFAULT ICI]
```

## 🛠️ Corrections des problèmes Qt

### Problème 1: Threading Qt

**Symptôme:**
```
QObject::setParent: Cannot set parent, new parent is in a different thread
```

**Cause:** Tentative de manipuler des objets Qt depuis un thread worker.

**Solution:** Les objets Qt doivent être créés et manipulés dans le thread principal (GUI thread).

### Problème 2: QPainter non fermé

**Symptôme:**
```
QBackingStore::endPaint() called with active painter
```

**Cause:** Un QPainter n'a pas été correctement fermé.

**Solution:** Toujours appeler `painter.end()` ou utiliser un context manager:
```python
with QPainter(widget) as painter:
    # draw something
    pass  # end() automatique
```

### Problème 3: Segfault

**Causes possibles:**
1. Accès à des objets Qt depuis un thread incorrect
2. Objets Qt détruits pendant leur utilisation
3. Problèmes de mémoire (CUDA/GPU)
4. Bibliothèques incompatibles

## 🔧 Mode debug avancé

### Activer Qt debug

```bash
export QT_DEBUG_PLUGINS=1
export QT_LOGGING_RULES='*.debug=true'
python3 run_with_logging.py
```

### Activer logging Python

```python
# Dans run_with_logging.py, ligne 58:
set_debug_mode(True)  # Déjà activé par défaut
```

### Utiliser gdb pour capturer le segfault

```bash
gdb python3
(gdb) run run_with_logging.py
# Attendre le segfault
(gdb) bt  # Afficher la stack trace
(gdb) quit
```

### Utiliser strace

```bash
strace -o trace.log python3 run_with_logging.py 2>&1
# Après le crash:
tail -200 trace.log
```

## 📝 Corrections apportées

### 1. API SAM3 compatible

Le backend détecte automatiquement l'API de SAM3:
- Nouvelle API: avec `load_from_HF`
- Ancienne API: sans `load_from_HF`

```python
try:
    model = build_sam3_image_model(..., load_from_HF=True, ...)
except TypeError:
    # Fallback sur ancienne API
    model = build_sam3_image_model(..., checkpoint_path=path, ...)
```

### 2. Logging complet

Tous les modules utilisent maintenant le système de logging:
- `SAM3Backend`: logs de toutes les opérations
- `MainWindow`: logs des opérations GUI
- `Worker`: logs des threads

### 3. Capture d'exceptions

Toutes les exceptions sont capturées et loggées avec stack trace complète.

## 🧪 Tests de débogage

### Test 1: Charger SAM3

```bash
python3 << 'EOF'
from sam3roto.backend.sam3_backend import SAM3Backend
backend = SAM3Backend()
backend.load("facebook/sam3-hiera-large")
print("✓ SAM3 chargé")
EOF
```

### Test 2: Ouvrir l'interface

```bash
python3 run_with_logging.py
# Vérifier que l'interface s'ouvre sans erreur
```

### Test 3: Charger une vidéo

```bash
# Dans l'interface:
# 1. Charger une vidéo de test (quelques frames)
# 2. Observer les logs
# 3. Si crash, analyser le dernier log
```

## 📊 Informations à fournir en cas de problème

Si vous rencontrez toujours des problèmes, fournissez:

1. **Le fichier de log complet:**
   ```bash
   cat ~/.sam3roto/logs/sam3roto_YYYYMMDD_HHMMSS.log
   ```

2. **Informations système:**
   ```bash
   python3 --version
   pip list | grep -E "torch|PySide6|sam3"
   nvidia-smi
   uname -a
   ```

3. **Stack trace du segfault** (si disponible avec gdb)

4. **Dernières lignes avant le crash:**
   ```bash
   tail -100 ~/.sam3roto/logs/sam3roto_YYYYMMDD_HHMMSS.log
   ```

## 🔗 Ressources

- [Qt Threading Documentation](https://doc.qt.io/qt-6/threads-qobject.html)
- [PySide6 Debugging](https://doc.qt.io/qtforpython-6/debugging.html)
- [Python Logging](https://docs.python.org/3/library/logging.html)

## 🚀 Prochaines étapes

1. Lancer avec `run_with_logging.py`
2. Reproduire le problème
3. Consulter les logs dans `~/.sam3roto/logs/`
4. Identifier la dernière opération avant le crash
5. Fournir les informations de débogage

---

**Mis à jour:** 2025-12-03
**Version:** 1.0
