# Guide d'Installation des Checkpoints SAM3 et DA3 en Local

## Problème
Si vous avez téléchargé les checkpoints SAM3 ou Depth Anything 3 manuellement et que vous voulez les utiliser sans les re-télécharger depuis HuggingFace, ce guide explique où les placer.

---

## 📦 Structure des Checkpoints

### SAM3 Checkpoints

Les checkpoints SAM3 doivent être dans un format compatible avec le repo officiel.

**Structure attendue** :
```
<chemin_checkpoint_sam3>/
├── config.json
├── model.safetensors (ou pytorch_model.bin)
├── preprocessor_config.json
└── (autres fichiers du modèle)
```

### Méthode 1 : Utiliser le Cache HuggingFace (Recommandé)

Si vous avez téléchargé depuis HuggingFace, placez les dans :

```bash
~/.cache/huggingface/hub/models--facebook--sam3-hiera-large/
```

**Structure complète** :
```
~/.cache/huggingface/hub/
└── models--facebook--sam3-hiera-large/
    ├── refs/
    ├── snapshots/
    │   └── <commit_hash>/
    │       ├── config.json
    │       ├── model.safetensors
    │       ├── preprocessor_config.json
    │       └── ...
    └── ...
```

L'application détectera automatiquement les fichiers ici quand vous entrez `facebook/sam3-hiera-large`.

### Méthode 2 : Chemin Absolu Local

Placez vos checkpoints n'importe où et référencez le chemin complet dans l'interface.

**Exemple** :
```bash
# Créer un dossier pour vos modèles
mkdir -p ~/models/sam3

# Placer les fichiers
cp /chemin/vers/vos/fichiers/* ~/models/sam3/

# Structure finale
~/models/sam3/
├── config.json
├── model.safetensors
├── preprocessor_config.json
└── ...
```

**Dans l'application** :
- Entrez le chemin complet : `/home/reepost/models/sam3`
- Cliquez "⚙️ Charger SAM3"

### Méthode 3 : Chemin Relatif dans le Projet

Placez les checkpoints dans le dossier du projet :

```bash
cd /home/reepost/Downloads/sam4-main
mkdir -p checkpoints/sam3
cp /chemin/vers/vos/fichiers/* checkpoints/sam3/
```

**Dans l'application** :
- Entrez : `checkpoints/sam3`
- Ou le chemin absolu : `/home/reepost/Downloads/sam4-main/checkpoints/sam3`

---

## 🌊 Depth Anything 3 Checkpoints

### Méthode 1 : Cache HuggingFace

```bash
~/.cache/huggingface/hub/models--depth-anything--DA3-LARGE/
```

### Méthode 2 : Dossier Local

```bash
mkdir -p ~/models/da3
# Copier les fichiers du modèle DA3
```

**Structure attendue** :
```
~/models/da3/
├── config.yaml (ou config.json)
├── model.safetensors (ou .pth)
└── ...
```

---

## 🔧 Vérification des Fichiers

### Pour SAM3

Vérifiez que vous avez au minimum :

```bash
ls -lh /chemin/vers/sam3/
# Doit contenir :
# config.json
# model.safetensors ou pytorch_model.bin
# preprocessor_config.json (optionnel)
```

### Pour DA3

```bash
ls -lh /chemin/vers/da3/
# Doit contenir :
# config.yaml ou config.json
# model.safetensors ou model.pth
```

---

## 🚀 Utilisation dans l'Application

### Charger SAM3

1. **Ouvrir l'application** : `python run.py`

2. **Entrer le chemin du modèle** dans le champ "SAM3 model id":
   - HuggingFace ID : `facebook/sam3-hiera-large` (cherche dans ~/.cache)
   - Chemin absolu : `/home/reepost/models/sam3`
   - Chemin relatif : `checkpoints/sam3`

3. **Cliquer "⚙️ Charger SAM3"**

4. **Vérifier le statut** : Doit afficher "✅ SAM3 chargé avec succès"

### Charger DA3

1. **Onglet "Depth / Camera (DA3)"**

2. **Entrer le model ID** :
   - HuggingFace : `depth-anything/DA3-LARGE`
   - Local : `/home/reepost/models/da3`

3. **Cliquer "⚙️ Charger DA3"**

---

## 🐛 Troubleshooting

### Erreur : "SAM3 n'est pas installé"

**Cause** : Le repo SAM3 n'est pas installé

**Solution** :
```bash
cd /home/reepost/Downloads/sam4-main
bash install_models.sh
```

Ou manuellement :
```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### Erreur : "FileNotFoundError" ou "Cannot load checkpoint"

**Cause** : Chemin incorrect ou fichiers manquants

**Solution** :
1. Vérifier que le chemin existe :
   ```bash
   ls -lh /chemin/vers/modele/
   ```

2. Vérifier la structure des fichiers :
   ```bash
   find /chemin/vers/modele/ -type f
   ```

3. Essayer le chemin absolu complet

### Erreur : "RuntimeError: CUDA out of memory"

**Cause** : GPU insuffisant

**Solutions** :
1. Utiliser un modèle plus petit :
   - `facebook/sam3-hiera-base` au lieu de `large`
   - `depth-anything/DA3-BASE` au lieu de `LARGE`

2. Réduire la résolution des images/vidéos

### Les erreurs Qt persistent (QThread, QBasicTimer)

**Cause** : Version ancienne du code

**Solution** : Pull les derniers changements avec le threading corrigé :
```bash
cd /home/reepost/Downloads/sam4-main
git pull origin claude/analyze-app-archive-01Qme7Y6vtqGVGRXBW2BwkKF
```

Ou re-télécharger :
```bash
cd /home/reepost/Downloads
rm -rf sam4-main
git clone https://github.com/nebegreg/sam4.git sam4-main
cd sam4-main
git checkout claude/analyze-app-archive-01Qme7Y6vtqGVGRXBW2BwkKF
```

---

## 📋 Checklist Complète

- [ ] Repo SAM3 installé : `pip list | grep sam3`
- [ ] Repo DA3 installé : `pip list | grep depth-anything`
- [ ] Checkpoints téléchargés et placés
- [ ] Chemins vérifiés avec `ls -lh`
- [ ] Code à jour (threading corrigé)
- [ ] Application lance sans erreurs Qt
- [ ] SAM3 se charge : "✅ SAM3 chargé avec succès"
- [ ] DA3 se charge : "✅ DA3 chargé"

---

## 💡 Recommandations

### Pour un usage optimal :

1. **Utiliser le cache HuggingFace** (~/.cache) si possible
   - Évite la duplication de fichiers
   - Compatible avec d'autres outils
   - Gestion automatique des versions

2. **Pour les checkpoints custom** :
   - Les placer dans `~/models/` avec structure claire
   - Utiliser des chemins absolus pour éviter confusion

3. **Tester d'abord avec des modèles petits** :
   - `sam3-hiera-base` (plus léger)
   - `DA3-BASE` (plus rapide)

4. **Vérifier l'espace disque** :
   - SAM3-large : ~2-3 GB
   - DA3-LARGE : ~1-2 GB
   - DA3NESTED-GIANT-LARGE : ~5 GB

---

## 📞 Support

Si vous avez toujours des problèmes après avoir suivi ce guide :

1. Vérifier les logs d'erreur dans le terminal
2. Partager l'erreur complète (avec traceback)
3. Vérifier que tous les prérequis sont installés :
   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import cv2; print(cv2.__version__)"
   python -c "from PySide6 import QtCore; print(QtCore.__version__)"
   ```

**Bonne chance ! 🚀**
