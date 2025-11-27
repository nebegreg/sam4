# 🚀 ROADMAP - SAM3 Roto Ultimate

**Vision**: Devenir l'outil de rotoscoping AI open-source de référence en 2025

---

## ✅ Phase 1: CORRECTIFS CRITIQUES (TERMINÉ)

**Status**: ✅ **COMPLÉTÉ - 27 Nov 2025**

- ✅ Corriger Sam3Processor API (set_text_prompt)
- ✅ Corriger interactive segmentation (add_geometric_prompt)
- ✅ Corriger extraction de masques
- ✅ Ajouter gestion d'erreurs video predictor
- ✅ Audit complet du code
- ✅ Documentation AUDIT_REPORT.md

**Résultat**: Application fonctionnelle avec SAM3 GitHub repo

---

## 🟡 Phase 2: AMÉLIORATIONS ESSENTIELLES (PRIORITÉ HAUTE)

**Timeline**: 1-2 semaines

### 2.1 Fallback SAM2
**Objectif**: Support immédiat sans installer repo GitHub

**Tasks**:
- [ ] Détecter si transformers a SAM2
- [ ] Implémenter backend SAM2 avec limitations documentées
- [ ] Fallback automatique SAM3 → SAM2 → Error
- [ ] Tests de compatibilité

**Bénéfices**:
- Installation ultra-simple (pip install transformers)
- Compatibilité immédiate
- Graceful degradation

---

### 2.2 Optical Flow avec RAFT
**Objectif**: Éliminer le flickering, cohérence temporelle

**Research**:
- **RAFT** (ECCV 2020): 44 FPS, SOTA optical flow
- 60% réduction artifacts vs Lucas-Kanade
- PyTorch implementation disponible

**Tasks**:
- [ ] Installer RAFT (pip install raft-pytorch)
- [ ] Créer `sam3roto/post/optical_flow.py`
- [ ] Intégrer flow warping dans temporal_smooth
- [ ] Flow-guided alpha refinement
- [ ] Benchmarks flickering before/after

**API Proposal**:
```python
def flow_based_temporal_smooth(
    masks: List[np.ndarray],
    frames: List[np.ndarray],
    method: str = "raft"  # raft, farneback
) -> List[np.ndarray]:
    """Apply optical flow guided temporal smoothing"""
```

**Bénéfices**:
- Élimination flickering
- Masques temporellement cohérents
- Qualité professionnelle

---

### 2.3 Amélioration Temporal Smoothing
**Objectif**: Remplacer simple moyenne par techniques SOTA

**Current**: Simple rolling average (obsolète 2020)

**Target**: Memory-based propagation (SOTA 2025)

**Tasks**:
- [ ] Rechercher MatAnyone implementation
- [ ] Consistent memory propagation
- [ ] Temporal attention mechanism
- [ ] Benchmark vs current approach

**References**:
- [MatAnyone Paper (Jan 2025)](https://arxiv.org/html/2501.14677v1)

---

## 🟢 Phase 3: FEATURES AVANCÉES (PRIORITÉ MOYENNE)

**Timeline**: 1-2 mois

### 3.1 Intégration MODNet
**Objectif**: Portrait matting real-time 67 FPS

**MODNet Features**:
- Trimap-free matting
- Real-time 67 FPS on 1080Ti
- Objective decomposition
- Portrait spécialisé

**Tasks**:
- [ ] pip install modnet
- [ ] Créer `sam3roto/backend/modnet_backend.py`
- [ ] UI selector: SAM3 / MODNet / RVM
- [ ] Preset "Portrait" utilisant MODNet
- [ ] Benchmarks performance

**Use Cases**:
- Video calls enhancement
- Portrait photography
- Real-time preview

---

### 3.2 Intégration RVM (Robust Video Matting)
**Objectif**: Video matting avec ConvGRU temporal consistency

**RVM Features**:
- ConvGRU recurrent architecture
- Temporal information utilization
- Background confusion reduction
- Designed specifically for video

**Tasks**:
- [ ] pip install rvm
- [ ] Backend RVM
- [ ] Comparison SAM3 vs RVM
- [ ] Hybrid mode: SAM3 detection + RVM refinement

**Use Cases**:
- Long videos
- Moving backgrounds
- Complex scenes

---

### 3.3 Batch Processing GPU
**Objectif**: 2-3x speedup via batching

**Current**: Frame-by-frame processing

**Target**: Batch processing with `set_image_batch`

**Tasks**:
- [ ] Utiliser Sam3Processor.set_image_batch()
- [ ] Dynamic batch sizing based on GPU memory
- [ ] Progress bar avec tqdm
- [ ] Benchmark speedup

**Expected Gains**:
- 2-3x faster GPU utilization
- Reduced overhead
- Better memory management

---

### 3.4 Cache Intelligent
**Objectif**: 3-5x speedup via réutilisation features

**Strategy**:
- LRU cache pour backbone features
- Réutilisation embeddings
- Propagation incrémentale

**Tasks**:
- [ ] Créer `sam3roto/cache/feature_cache.py`
- [ ] LRU eviction policy
- [ ] Disk persistence (optional)
- [ ] Memory usage monitoring

**Expected Gains**:
- 3-5x speedup sur vidéos similaires
- Reduced model calls
- Interactive performance

---

## 🔵 Phase 4: POLISH PROFESSIONNEL (PRIORITÉ BASSE)

**Timeline**: 2-3 mois

### 4.1 Exports Professionnels
**Objectif**: Formats industry-standard

**Formats à ajouter**:
- **EXR 32-bit**: VFX standard, HDR support
- **DPX**: Cinéma, log encoding
- **Cryptomatte**: ID mattes pour Nuke/After Effects
- **SGI RGB**: Legacy compositing

**Tasks**:
- [ ] OpenEXR integration (pip install OpenEXR)
- [ ] DPX writer avec correct color space
- [ ] Cryptomatte manifest generation
- [ ] Tests avec Nuke/After Effects

**Use Cases**:
- Film production
- VFX pipelines
- Compositing workflows

---

### 4.2 Intégrations Professionnelles
**Objectif**: Interopérabilité avec outils existants

**Integrations**:
- **Version Zero AI**: Splines output API
- **Mocha Pro**: Tracking data import
- **Nuke**: Gizmo/plugin
- **After Effects**: ExtendScript plugin

**Tasks**:
- [ ] Rechercher APIs disponibles
- [ ] Créer exporters spécifiques
- [ ] Documentation integration
- [ ] Validation avec pros

---

### 4.3 Interface Améliorée
**Objectif**: UX professionnelle

**Features**:
- **Timeline Professional**:
  - Keyframe editor
  - Bezier curves
  - Onion skinning

- **Brush Tool**:
  - Paint to refine
  - Pressure sensitivity (Wacom)
  - Undo/Redo stack

- **Preset Manager**:
  - Save/Load custom presets
  - Share community presets
  - Auto-detect scene type

**Tasks**:
- [ ] Redesign UI avec Qt Designer
- [ ] Implement custom widgets
- [ ] Shortcuts configurables
- [ ] Dark/Light themes

---

### 4.4 Documentation Interactive
**Objectif**: Adoption massive

**Content**:
- **Jupyter Notebooks**:
  - Step-by-step tutorials
  - Interactive examples
  - Live preview

- **Video Tutorials**:
  - YouTube series
  - Use case demonstrations
  - Tips & tricks

- **Community**:
  - Discord server
  - Gallery showcase
  - Feature requests

**Tasks**:
- [ ] Créer notebooks/ directory
- [ ] Enregistrer video tutorials
- [ ] Setup Discord
- [ ] Create gallery website

---

## 💎 Phase 5: RECHERCHE & INNOVATION (LONG TERME)

**Timeline**: 6+ mois

### 5.1 Diffusion-Based Matting
**Trend**: Generative models pour matting

**Research**:
- Generative Video Matting (Aug 2025)
- Diffusion models pour alpha
- Text-to-matte generation

**POC**:
- [ ] Survey diffusion matting papers
- [ ] Implement prototype
- [ ] Compare vs traditional

---

### 5.2 NeRF Integration
**Vision**: 3D-aware matting

**Use Cases**:
- Multi-view consistency
- 3D reconstruction
- Novel view synthesis

**Research**:
- NeRF + matting papers
- Gaussian Splatting integration
- Real-time rendering

---

### 5.3 On-Device Edge Deployment
**Objective**: Run on mobile/edge devices

**Targets**:
- CoreML (iOS)
- TensorFlow Lite (Android)
- ONNX Runtime (cross-platform)

**Tasks**:
- [ ] Model quantization
- [ ] Mobile optimization
- [ ] Real-time preview
- [ ] App prototypes

---

## 📊 METRICS & SUCCESS CRITERIA

### Performance Targets

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Flickering | ~15% | <5% | RAFT optical flow |
| FPS (1080p) | ~5 FPS | >10 FPS | Batch + Cache |
| Memory | 8GB | <6GB | Efficient caching |
| Quality (IOU) | ~0.85 | >0.90 | Better matting |

### Adoption Targets

| Metric | 3 Months | 6 Months | 12 Months |
|--------|----------|----------|-----------|
| GitHub Stars | 100 | 500 | 2000 |
| Active Users | 50 | 300 | 1500 |
| Community Presets | 10 | 50 | 200 |

---

## 🎯 PRIORITIZATION MATRIX

### Must Have (Phase 1-2)
- ✅ Core functionality working
- 🟡 SAM2 fallback
- 🟡 Temporal consistency (RAFT)

### Should Have (Phase 3)
- MODNet integration
- RVM integration
- Batch processing
- Cache system

### Nice to Have (Phase 4)
- Pro exports (EXR, DPX)
- Advanced UI
- Documentation

### Research (Phase 5)
- Diffusion matting
- NeRF integration
- Edge deployment

---

## 📚 RESOURCES & REFERENCES

### Papers
- [SAM 3 (Meta, Nov 2025)](https://github.com/facebookresearch/sam3)
- [MatAnyone (Jan 2025)](https://arxiv.org/html/2501.14677v1)
- [Generative Video Matting (Aug 2025)](https://arxiv.org/html/2508.07905v1)
- [RAFT Optical Flow (ECCV 2020)](https://arxiv.org/abs/2003.12039)
- [MODNet (AAAI 2022)](https://arxiv.org/abs/2011.11961)
- [RVM Robust Video Matting](https://arxiv.org/abs/2108.11515)

### Tools
- [Boris FX Mocha Pro](https://borisfx.com/products/mocha-pro/)
- [Version Zero AI](https://www.versionzero.ai/)
- [Mask Prompter](https://github.com/mask-prompter)

### Communities
- Reddit: r/VFX, r/computervision
- Discord: PyTorch, OpenCV
- Twitter: #ComputerVision, #VFX

---

## 🤝 CONTRIBUTION GUIDELINES

### How to Contribute

1. **Pick a Task** from Phase 2-3
2. **Create Issue** describing your approach
3. **Fork & Branch** from main
4. **Implement** with tests
5. **Submit PR** with detailed description
6. **Iterate** based on feedback

### Code Standards

- **Type hints** required
- **Docstrings** for all public APIs
- **Tests** for new features
- **No TODOs** in committed code

### Review Process

- All PRs reviewed within 48h
- CI must pass (tests, linting)
- Performance benchmarks required for optimizations
- Documentation updates mandatory

---

## 💬 FEEDBACK & QUESTIONS

**Questions?** Open a GitHub Discussion

**Bugs?** File an Issue with reproducible steps

**Feature Ideas?** Comment on ROADMAP or create RFC

**Need Help?** Join our Discord (coming soon)

---

**Last Updated**: 2025-11-27
**Version**: 1.0
**Maintainer**: @claude-code-ultimate
