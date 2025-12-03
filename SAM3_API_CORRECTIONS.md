# Correction de l'API SAM3 - Rapport

## 🔍 Problème identifié

Le backend utilisait une **API incorrecte** qui mélangeait SAM2 et SAM3. Après recherche sur la [documentation officielle SAM3](https://github.com/facebookresearch/sam3), j'ai identifié plusieurs erreurs.

## ❌ Erreurs corrigées

### 1. Type de requête "propagate" incorrect

**Avant (incorrect):**
```python
response = self._video_predictor.handle_request(
    request=dict(
        type="propagate",  # ❌ N'existe pas dans SAM3
        session_id=session_id,
    )
)
```

**Après (correct):**
```python
for response in self._video_predictor.handle_stream_request(
    request=dict(
        type="propagate_in_video",  # ✅ Nom correct
        session_id=session_id,
    )
):
    # Traite chaque frame au fur et à mesure
```

### 2. Mauvaise méthode pour la propagation

**Avant:** Utilisait `handle_request()` qui attend une réponse unique
**Après:** Utilise `handle_stream_request()` qui retourne un **générateur** frame par frame

### 3. Type de session "end_session" incorrect

**Avant (incorrect):**
```python
type="end_session"  # ❌ N'existe pas
```

**Après (correct):**
```python
type="close_session"  # ✅ Nom correct
```

### 4. Structure de réponse incorrecte

**Avant:** Récupérait tout dans un gros dictionnaire `outputs`
**Après:** Traite chaque frame individuellement via le stream

## ✅ API SAM3 Officielle

D'après la [documentation officielle](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_video_predictor_example.ipynb):

### Types de requêtes supportés

| Type | Méthode | Description |
|------|---------|-------------|
| `start_session` | `handle_request()` | Démarre une session vidéo |
| `add_prompt` | `handle_request()` | Ajoute un prompt (texte/points) |
| `propagate_in_video` | `handle_stream_request()` | Propage dans toute la vidéo |
| `reset_session` | `handle_request()` | Réinitialise la session |
| `close_session` | `handle_request()` | Ferme la session |
| `remove_object` | `handle_request()` | Supprime un objet tracké |

### Différences handle_request vs handle_stream_request

- **`handle_request()`**: Retourne une réponse unique (dict)
- **`handle_stream_request()`**: Retourne un générateur qui yield chaque frame

### Workflow correct

```python
# 1. Démarrer session
response = predictor.handle_request(
    request=dict(type="start_session", resource_path="/path/to/video")
)
session_id = response["session_id"]

# 2. Ajouter prompts
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text="person"
    )
)

# 3. Propager (streaming)
for response in predictor.handle_stream_request(
    request=dict(type="propagate_in_video", session_id=session_id)
):
    frame_idx = response["frame_index"]
    outputs = response["outputs"]

    for obj_id, obj_output in outputs.items():
        mask = obj_output["mask"]
        # Traiter le mask

# 4. Fermer session
predictor.handle_request(
    request=dict(type="close_session", session_id=session_id)
)
```

## 📊 Changements dans le code

### Fichier: `sam3roto/backend/sam3_backend.py`

#### Méthode `track_concept_video()`:
- ✅ Remplacé `type="propagate"` par `type="propagate_in_video"`
- ✅ Remplacé `handle_request()` par `handle_stream_request()`
- ✅ Remplacé `type="end_session"` par `type="close_session"`
- ✅ Correction extraction des masks: `obj_output.get("mask")` au lieu de `outputs[frame_idx]["masks"]`
- ✅ Ajout de logs détaillés pour chaque frame

#### Méthode `track_interactive_video()`:
- ✅ Mêmes corrections que ci-dessus
- ✅ Correction des paramètres de prompts: `points`, `labels`, `object_id`

## 🎯 Impact des corrections

### Avant (ne fonctionnait pas):
```
[SAM3 Video] Propagating through video...
AttributeError: 'Sam3Model' object has no attribute 'handle_request'
```

### Après (devrait fonctionner):
```
[SAM3 Video] Propagating through video...
[SAM3 Video] Frame 0 response
[SAM3 Video] Frame 0: 1 objects
[SAM3 Video] Frame 1 response
[SAM3 Video] Frame 1: 1 objects
...
[SAM3 Video] Closing session abc123...
[SAM3 Video] Session closed successfully
```

## 📚 Sources consultées

- [SAM3 GitHub Repository](https://github.com/facebookresearch/sam3)
- [SAM3 Video Predictor Example](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_video_predictor_example.ipynb)
- [SAM3 Video API Documentation](https://deepwiki.com/facebookresearch/sam3/4.1-video-api-usage)
- [SAM3 Hugging Face](https://huggingface.co/facebook/sam3)

## ⚠️ Notes importantes

1. **Installation requise**: Le repo GitHub SAM3 DOIT être installé
   ```bash
   cd /tmp && git clone https://github.com/facebookresearch/sam3.git
   cd sam3 && pip install -e .
   ```

2. **Transformers ne suffit pas**: La version transformers n'a PAS l'API vidéo complète

3. **Streaming obligatoire**: `propagate_in_video` retourne un générateur, pas un dict complet

## 🔄 Migration

Si vous utilisiez l'ancienne API:

| Ancien code | Nouveau code |
|-------------|--------------|
| `type="propagate"` | `type="propagate_in_video"` |
| `type="end_session"` | `type="close_session"` |
| `handle_request()` pour propagate | `handle_stream_request()` |
| `response["outputs"][frame_idx]` | `response["outputs"]` (déjà filtré par frame) |

## ✅ Prochaines étapes

1. ✅ Corriger l'API SAM3 (fait)
2. 🔄 Installer SAM3 GitHub
3. 🔄 Tester le tracking vidéo
4. 🔄 Simplifier le GUI (à venir)

---

**Date:** 2025-12-03
**Version:** 2.0
**Status:** ✅ API corrigée selon la documentation officielle
