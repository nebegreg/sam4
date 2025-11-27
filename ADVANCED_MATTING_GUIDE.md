# Guide du Matting Avancé - SAM3 Roto Ultimate

Ce guide explique les nouvelles fonctionnalités avancées de matting pour obtenir des masques parfaits, notamment pour les cheveux, fourrure et autres détails fins.

## 📚 Table des Matières

- [Nouveautés](#nouveautés)
- [Presets de Matting](#presets-de-matting)
- [Techniques Avancées](#techniques-avancées)
- [Workflows Recommandés](#workflows-recommandés)
- [Références Scientifiques](#références-scientifiques)

---

## 🆕 Nouveautés

### Matting Avancé Intégré

L'application intègre maintenant des techniques state-of-the-art de matting basées sur des recherches académiques :

1. **Guided Filter** - Pour raffiner les alphas en préservant les structures fines ([He et al., ECCV 2010](http://kaiminghe.com/eccv10/))
2. **Trimap Automatique Avancé** - Génération intelligente de trimaps ([Liu et al., 2017](https://arxiv.org/abs/1707.00333))
3. **Multi-Scale Refinement** - Capture les détails à différentes échelles
4. **Edge-Aware Smoothing** - Lissage guidé par l'image RGB

### Presets Optimisés

8 presets professionnels pour différents matériaux :
- Hair - Fine Details (cheveux fins, baby hair)
- Hair - Thick/Curly (cheveux épais, bouclés)
- Fur/Animal Hair (fourrure animale)
- Smoke/Fog (fumée, brouillard)
- Glass/Transparent (verre, objets transparents)
- Sharp Edges (logos, graphiques)
- Fabric/Clothing (tissus, vêtements)
- Motion Blur (objets en mouvement)

---

## 🎨 Presets de Matting

### Comment Utiliser les Presets

1. Dans l'onglet **Matte**, section **Matting Presets**
2. Sélectionner un preset dans le menu déroulant
3. Les paramètres se chargent automatiquement
4. Ajuster finement si nécessaire (le preset passe en "Custom")

### Détails des Presets

#### 🌟 Hair - Fine Details
**Quand l'utiliser** : Cheveux fins, mèches, baby hair, détails capillaires très fins

**Paramètres clés** :
- Trimap band: 20px (zone de transition large)
- Advanced matting: Both (Guided + Trimap)
- Multi-scale: Activé (capture différentes échelles)
- Guided radius: 8px avec eps=1e-5 (très précis)
- Temporal smooth: 65% (stabilité élevée)

**Résultats** :
- ✅ Préserve les mèches très fines
- ✅ Bords doux et naturels
- ✅ Aucun artefact sur les cheveux transparents
- ⚠️ Plus lent (multi-scale)

#### 💇 Hair - Thick/Curly
**Quand l'utiliser** : Cheveux épais, bouclés, afro, dreadlocks

**Paramètres clés** :
- Trimap band: 15px
- Advanced matting: Guided Filter uniquement
- Multi-scale: Désactivé (plus rapide)
- Fill holes: 200px (comble les espaces entre boucles)
- Temporal smooth: 70% (très stable)

**Résultats** :
- ✅ Excellent pour cheveux volumineux
- ✅ Traitement rapide
- ✅ Bonne gestion des boucles serrées

#### 🦊 Fur/Animal Hair
**Quand l'utiliser** : Fourrure d'animaux, pelage, poils d'animaux

**Paramètres clés** :
- Trimap band: 25px (très large pour les poils qui dépassent)
- Guided radius: 12px (structure plus large)
- Edge-aware smoothing activé
- Pixel spread: 15px (étend bien les bords)

**Résultats** :
- ✅ Capture les poils individuels
- ✅ Préserve la texture de la fourrure
- ✅ Excellent pour gros plans d'animaux

#### 💨 Smoke/Fog
**Quand l'utiliser** : Fumée, brouillard, vapeur, effets atmosphériques

**Paramètres clés** :
- Feather: 8px (bords très doux)
- Trimap: Désactivé (pas de bords nets)
- Guided eps: 0.001 (lissage élevé)
- Temporal smooth: 75% (stabilité maximale)
- Despill strength: 60% (doux pour préserver les couleurs)

**Résultats** :
- ✅ Alphas semi-transparents naturels
- ✅ Pas d'artefacts de bord
- ✅ Excellent pour éléments volumineux

#### 🔍 Glass/Transparent
**Quand l'utiliser** : Verre, lunettes, bouteilles, objets transparents, reflets

**Paramètres clés** :
- Guided eps: 1e-6 (ultra précis pour les reflets)
- Trimap band: 10px (petit)
- Multi-scale activé
- Edge-aware avec sigma faible (5.0)

**Résultats** :
- ✅ Préserve les reflets et refractions
- ✅ Alphas partiels précis
- ✅ Bon pour compositing complexe

#### ✏️ Sharp Edges
**Quand l'utiliser** : Logos, graphiques, texte, objets avec bords nets

**Paramètres clés** :
- Feather: 0.5px (minimal)
- Fill holes: 500px (comble les zones solides)
- Advanced matting: Désactivé
- Border fix: 3px

**Résultats** :
- ✅ Bords nets et précis
- ✅ Pas de flou indésirable
- ✅ Traitement très rapide

#### 👕 Fabric/Clothing
**Quand l'utiliser** : Vêtements, tissus, robes, chemises

**Paramètres clés** :
- Trimap band: 12px
- Guided filter activé
- Fill holes: 300px (gère les plis)
- Pixel spread: 10px

**Résultats** :
- ✅ Bonne gestion des textures
- ✅ Préserve les plis et détails
- ✅ Équilibré vitesse/qualité

#### 🏃 Motion Blur
**Quand l'utiliser** : Objets en mouvement rapide, flou de bougé

**Paramètres clés** :
- Grow/shrink: +2px (compense le flou)
- Feather: 5px (bords flous)
- Temporal smooth: 80% (stabilité maximale)
- Edge-aware activé

**Résultats** :
- ✅ Suit le mouvement naturellement
- ✅ Préserve le flou de bougé
- ✅ Cohérence temporelle élevée

---

## 🔬 Techniques Avancées

### Guided Filter

Le **Guided Filter** ([He et al., ECCV 2010](http://kaiminghe.com/eccv10/)) est une technique de filtrage edge-preserving qui utilise l'image RGB comme guide pour raffiner l'alpha.

**Quand l'utiliser** :
- Cheveux, fourrure avec structure fine
- Quand le masque initial est bon mais a besoin de raffinement
- Pour préserver les détails tout en lissant le bruit

**Paramètres** :
- `radius`: Rayon du filtre (8-12px pour cheveux)
- `eps` (epsilon): Régularisation (1e-5 pour précis, 1e-3 pour lissé)

**Formule** :
```
alpha_raffiné = a * RGB + b
où a et b sont calculés pour suivre les edges de RGB
```

### Trimap Automatique Avancé

Génération automatique d'un **trimap** (Foreground/Unknown/Background) optimisée pour les détails fins ([Liu et al., 2017](https://arxiv.org/abs/1707.00333)).

**Avantages** :
- Définit précisément les zones incertaines (cheveux, bords flous)
- Permet aux algorithmes de matting de se concentrer sur les zones difficiles
- Réduit les artefacts sur les bords

**Zone Unknown** :
```
Unknown = (Masque dilaté) - (Masque érodé)
```

### Multi-Scale Refinement

Traite l'image à plusieurs échelles (100%, 50%, 25%) pour capturer :
- **Grande échelle** : Structure globale
- **Échelle moyenne** : Formes intermédiaires
- **Petite échelle** : Détails fins (cheveux individuels)

**Trade-off** :
- ✅ Qualité maximale
- ⚠️ 2-3x plus lent
- **Recommandé pour** : Cheveux fins, fourrure, exports finaux

### Edge-Aware Smoothing

Lissage bilatéral guidé par l'image RGB qui :
- Lisse l'alpha dans les zones uniformes
- Préserve les edges de l'image RGB
- Élimine le bruit sans flouter les détails

**Formule** :
```
alpha_smooth[p] = Σ G_σs(||p-q||) * G_σc(||RGB[p]-RGB[q]||) * alpha[q]
```

---

## 🎬 Workflows Recommandés

### Workflow 1 : Cheveux Fins (Qualité Maximale)

1. **Segmentation SAM3**
   - Mode PCS avec prompt "person" ou "hair"
   - Ou mode PVS avec quelques points sur les cheveux

2. **Preset**
   - Sélectionner "Hair - Fine Details"

3. **Ajustements**
   - Si trop de bruit : augmenter `Temporal smooth` à 70-75%
   - Si perte de détails : réduire `Trimap band` à 15-18px
   - Si cheveux trop transparents : activer uniquement "Guided Filter"

4. **Preview**
   - Vérifier avec overlay mode "Contour Only"
   - Utiliser "Checkerboard" pour voir la transparence

5. **Export**
   - Export RGBA straight (pas premultiplied)
   - Utiliser "Edge extend / Pixel spread" (10-12px)

### Workflow 2 : Fourrure Animale

1. **Segmentation**
   - PCS: "animal fur" ou "dog" / "cat"
   - Ou PVS avec box sur l'animal

2. **Preset**
   - "Fur/Animal Hair"

3. **Ajustements**
   - Pour poils très fins : activer "Multi-scale refinement"
   - Pour fourrure dense : désactiver multi-scale (plus rapide)
   - Augmenter `Pixel spread` à 15-20px si halos noirs

4. **Despill**
   - Utiliser "Physical (auto BG)" mode
   - Strength: 70-80%
   - Luminance restore: activé

### Workflow 3 : Fumée / Éléments Semi-Transparents

1. **Segmentation**
   - PCS: "smoke" / "fog" / "steam"
   - Points + aux centres denses

2. **Preset**
   - "Smoke/Fog"

3. **Ajustements**
   - Désactiver "Fill holes" (laisse les zones transparentes)
   - Augmenter `Feather` à 10-15px pour bords très doux
   - Temporal smooth à 80% minimum

4. **RGB Cleanup**
   - Despill doux (50-60%)
   - Pas de pixel spread (garde la transparence)
   - Export premultiplied

### Workflow 4 : Tracking Vidéo avec Raffinement

1. **Tracking SAM3**
   - PCS vidéo ou PVS vidéo avec keyframes
   - Laisse SAM3 générer les masques initiaux

2. **Premier Pass**
   - Preset "Default/Balanced"
   - Preview rapidement toute la séquence

3. **Identifier les Problèmes**
   - Frames avec perte de détails → utiliser "Hair Fine"
   - Frames avec bruit → augmenter temporal smooth
   - Frames avec trous → augmenter fill holes

4. **Second Pass Ciblé**
   - Re-segmenter les frames problématiques
   - Utiliser preset approprié
   - Vérifier cohérence temporelle

5. **Export Final**
   - Toujours vérifier quelques frames random
   - Export avec tous les RGB cleanups activés

---

## 📖 Références Scientifiques

### Guided Filter
- **Paper**: "Guided Image Filtering" - K. He, J. Sun, X. Tang (ECCV 2010)
- **URL**: http://kaiminghe.com/eccv10/
- **GitHub**: https://github.com/atilimcetin/guided-filter

### Alpha Matting
- **Deep Image Matting**: https://arxiv.org/abs/1703.03872
- **GCA-Matting**: https://arxiv.org/abs/2001.04069 | [GitHub](https://github.com/Yaoyi-Li/GCA-Matting)
- **Trimap Generation**: https://arxiv.org/abs/1707.00333

### Video Matting
- **MODNet** (Real-Time Portrait Matting): [GitHub](https://github.com/ZHKKKe/MODNet)
- **Background Matting V2**: [GitHub](https://github.com/PeterL1n/BackgroundMattingV2)

### OpenCV Matting
- **Information Flow Matting**: https://docs.opencv.org/4.x/dd/d0e/tutorial_alphamat.html

### Industry Best Practices
- **Adobe Roto Brush 3**: [Rotoscoping Hair](https://helpx.adobe.com/after-effects/using/roto-brush-refine-matte.html)
- **FXGuide Roto**: https://www.fxguide.com/fxfeatured/the-art-of-roto-2011/
- **Video Segmentation UX**: https://www.v7labs.com/blog/video-segmentation-guide

---

## 💡 Tips & Astuces

### Pour les Cheveux

1. **Toujours utiliser l'image RGB** : Les algorithmes ont besoin de l'image originale pour guider le raffinement
2. **Ne pas sur-lisser** : Les cheveux ont naturellement du bruit, trop lisser les fait paraître plastiques
3. **Multi-scale pour exports finaux** : Active uniquement pour le résultat final, pas pendant les tests
4. **Trimap band adaptatif** : Cheveux courts = 10-15px, cheveux longs = 20-30px

### Pour la Performance

1. **Désactiver multi-scale** pendant les tests
2. **Utiliser "Guided Filter" seul** au lieu de "Both" pour 2x plus rapide
3. **Réduire temporal smooth** en dessous de 60% pour temps réel
4. **Preview avec lower resolution** si possible

### Pour le Compositing

1. **Toujours faire edge extend / pixel spread** avant export (élimine halos noirs)
2. **Despill physique** donne les meilleurs résultats mais est plus lent
3. **Luminance restore** critique pour éviter la désaturation
4. **Export straight alpha** pour plus de flexibilité en post

### Debugging

**Alpha trop transparent** :
- Augmenter `Guided eps` (1e-4 ou 1e-3)
- Réduire `Trimap band`
- Essayer "Guided Filter" au lieu de "Both"

**Trop de bruit sur les bords** :
- Activer "Edge-aware smoothing"
- Augmenter `Temporal smooth`
- Réduire `Trimap band`

**Perte de détails fins** :
- Réduire `Guided eps` (1e-5 ou 1e-6)
- Activer "Multi-scale refinement"
- Augmenter `Trimap band`

**Artefacts temporels (flickering)** :
- Augmenter `Temporal smooth` à 70-80%
- Utiliser preset "Motion Blur"
- Vérifier que SAM3 tracking est stable

---

## 🎯 Comparaison des Modes

| Mode | Qualité | Vitesse | Cas d'Usage |
|------|---------|---------|-------------|
| **Guided Filter seul** | ⭐⭐⭐⭐ | ⚡⚡⚡ | Cheveux épais, tissus |
| **Trimap seul** | ⭐⭐⭐ | ⚡⚡⚡ | Sharp edges, logos |
| **Both (Guided + Trimap)** | ⭐⭐⭐⭐⭐ | ⚡⚡ | Cheveux fins, fourrure |
| **Multi-scale** | ⭐⭐⭐⭐⭐ | ⚡ | Exports finaux uniquement |

---

**Bonne chance pour vos rotoscopies ! 🎬✨**
