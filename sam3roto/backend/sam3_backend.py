from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Any
import numpy as np
import torch
from PIL import Image
import tempfile
import shutil
from pathlib import Path
import sys

# Import logging
try:
    from ..utils.logging import get_logger
    logger = get_logger("SAM3Backend")
except ImportError:
    import logging
    logger = logging.getLogger("SAM3Backend")
    logger.setLevel(logging.INFO)

# Import optimizations
try:
    from ..utils import (
        get_memory_manager,
        get_feature_cache,
        timed_operation,
        torch_inference_mode,
    )
    _HAS_OPTIMIZATIONS = True
except ImportError:
    _HAS_OPTIMIZATIONS = False
    print("[SAM3Backend] Warning: Optimizations not available")

@dataclass
class FrameMasks:
    frame_idx: int
    masks_by_id: Dict[int, np.ndarray]  # obj_id -> mask_u8 (0/255)

def _mask_to_u8(mask) -> np.ndarray:
    # mask may be torch Tensor in [0,1] or bool
    if hasattr(mask, "detach"):
        mask = mask.detach().float().cpu().numpy()
    mask = np.asarray(mask)
    # ensure HxW
    if mask.ndim == 4:
        mask = mask[0,0]
    elif mask.ndim == 3:
        mask = mask[0]
    if mask.max() <= 1.0:
        mask = (mask > 0.5).astype(np.uint8) * 255
    else:
        mask = (mask > 0).astype(np.uint8) * 255
    return mask.astype(np.uint8)

class SAM3Backend:
    """Backend SAM3 basé sur le repo officiel GitHub facebook/sam3.

    - PCS image: Sam3Processor avec text prompts
    - PVS image: Sam3Processor avec point/box prompts
    - PCS vidéo: Sam3VideoPredictor avec text prompts
    - PVS vidéo: Sam3VideoPredictor avec point/box prompts

    Références:
    https://github.com/facebookresearch/sam3
    """
    def __init__(self, enable_optimizations: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16 if self.device.type == "cuda" else torch.float32

        self.model_id = "facebook/sam3-hiera-large"
        self._image_model = None
        self._image_processor = None
        self._video_predictor = None
        self._use_transformers = False  # Flag pour savoir quelle méthode est utilisée

        # Optimizations
        self.enable_optimizations = enable_optimizations and _HAS_OPTIMIZATIONS
        if self.enable_optimizations:
            self.memory_manager = get_memory_manager()
            self.feature_cache = get_feature_cache()
            print("[SAM3Backend] Optimizations ENABLED (memory management + caching)")
        else:
            self.memory_manager = None
            self.feature_cache = None
            print("[SAM3Backend] Optimizations DISABLED")

    def load(self, model_id_or_path: str = "facebook/sam3-hiera-large"):
        """Charge les modèles SAM3 pour image et vidéo.

        Essaie deux approches:
        1. Transformers/HuggingFace (Sam3Model, Sam3Processor)
        2. Repo GitHub officiel (build_sam3_image_model, build_sam3_video_predictor)
        """
        import traceback
        import sys

        # Print memory stats before loading
        if self.enable_optimizations:
            print("\n[SAM3] Memory before loading:")
            stats_before = self.memory_manager.get_stats()
            print(stats_before)

            # Check if we have enough memory
            estimated_model_size = 4.0  # GB estimate for SAM3 large
            device_str = "cuda" if self.device.type == "cuda" else "cpu"

            if not self.memory_manager.can_load_model(estimated_model_size, device=device_str):
                print(f"[SAM3] ⚠️  Warning: May not have enough memory! Attempting cleanup...")
                self.memory_manager.cleanup(aggressive=True)

        print(f"\n[SAM3] Début du chargement...")
        print(f"[SAM3] Device: {self.device}, dtype: {self.dtype}")
        print(f"[SAM3] Model ID/Path: {model_id_or_path}")

        # MÉTHODE 1: Essayer repo GitHub officiel (PRIORITAIRE pour support vidéo complet)
        # Cette méthode supporte l'API vidéo complète avec handle_request()
        try:
            print("\n[SAM3] 🔄 Tentative 1: Repo GitHub officiel...")
            from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
            from sam3.model.sam3_image_processor import Sam3Processor as Sam3ProcessorOfficial
            print("[SAM3] ✓ Imports repo GitHub réussis")

            self.model_id = model_id_or_path
            self._use_transformers = False

            # Déterminer si on charge depuis HuggingFace ou depuis un chemin local
            from pathlib import Path
            is_local_path = Path(model_id_or_path).exists()
            load_from_hf = not is_local_path

            print(f"[SAM3] Mode de chargement: {'HuggingFace' if load_from_hf else 'Local'}")

            # Image model + processor
            print(f"[SAM3] Chargement image model...")

            # Essayer avec load_from_HF (nouvelle API)
            try:
                self._image_model = build_sam3_image_model(
                    checkpoint_path=model_id_or_path if not load_from_hf else None,
                    load_from_HF=load_from_hf,
                    device=str(self.device),
                    eval_mode=True
                )
            except TypeError:
                # Fallback: ancienne API sans load_from_HF
                print("[SAM3] API ancienne détectée, essai sans load_from_HF...")
                self._image_model = build_sam3_image_model(
                    checkpoint_path=model_id_or_path if not load_from_hf else model_id_or_path,
                    device=str(self.device),
                    eval_mode=True
                )

            print(f"[SAM3] Image model construit (déjà sur {self.device})")

            # Le modèle est déjà sur le device, mais on peut changer le dtype
            if self.dtype in (torch.float16, torch.bfloat16):
                print(f"[SAM3] Conversion en {self.dtype}...")
                self._image_model = self._image_model.to(dtype=self.dtype)

            print("[SAM3] Création du processor...")
            self._image_processor = Sam3ProcessorOfficial(self._image_model)
            print("[SAM3] ✅ Image model OK")

            # Video predictor
            print(f"[SAM3] Chargement video predictor...")

            # Essayer avec load_from_HF (nouvelle API)
            try:
                self._video_predictor = build_sam3_video_predictor(
                    checkpoint_path=model_id_or_path if not load_from_hf else None,
                    load_from_HF=load_from_hf,
                    device=str(self.device)
                )
            except TypeError:
                # Fallback: ancienne API sans load_from_HF
                print("[SAM3] API ancienne détectée, essai sans load_from_HF...")
                self._video_predictor = build_sam3_video_predictor(
                    checkpoint_path=model_id_or_path if not load_from_hf else model_id_or_path,
                    device=str(self.device)
                )

            print(f"[SAM3] Video predictor construit")

            # Appliquer dtype si nécessaire
            if self.dtype in (torch.float16, torch.bfloat16):
                print(f"[SAM3] Conversion en {self.dtype}...")
                self._video_predictor.model = self._video_predictor.model.to(dtype=self.dtype)

            print("[SAM3] ✅ Video predictor OK")

            print("✅ SAM3 chargé avec succès (repo GitHub)")

            # Print memory stats after loading
            if self.enable_optimizations:
                print("\n[SAM3] Memory after loading:")
                stats_after = self.memory_manager.get_stats()
                print(stats_after)

            return

        except ImportError as e:
            print(f"[SAM3] ⚠️  Repo GitHub non disponible: {e}")
            print("[SAM3] Tentative de fallback sur transformers...")
        except Exception as e:
            print(f"[SAM3] ⚠️  Échec repo GitHub: {e}")
            print("[SAM3] Tentative de fallback sur transformers...")
            traceback.print_exc()

        # MÉTHODE 2: Fallback sur Transformers/HuggingFace (IMAGE SEULEMENT)
        # ⚠️ ATTENTION: La vidéo ne fonctionnera PAS avec transformers
        # L'API transformers n'a pas handle_request() nécessaire pour le tracking vidéo
        try:
            print("\n[SAM3] 🔄 Tentative 2: Transformers/HuggingFace (fallback)...")
            from transformers import Sam3Model, Sam3Processor
            print("[SAM3] ✓ Imports transformers réussis")

            self.model_id = model_id_or_path

            print(f"[SAM3] Chargement depuis transformers: {model_id_or_path}")
            model = Sam3Model.from_pretrained(model_id_or_path).to(self.device)
            processor = Sam3Processor.from_pretrained(model_id_or_path)

            if self.dtype in (torch.float16, torch.bfloat16):
                print(f"[SAM3] Conversion en {self.dtype}...")
                model = model.to(dtype=self.dtype)
            model.eval()

            # Stocker dans les attributs
            # ⚠️ IMPORTANT: On utilise le même modèle pour image ET vidéo,
            # mais la vidéo ne fonctionnera PAS (pas de handle_request)
            self._image_model = model
            self._image_processor = processor
            self._video_predictor = model  # Sera cassé pour la vidéo!
            self._use_transformers = True

            print("✅ SAM3 chargé avec succès (transformers)")
            print("⚠️  AVERTISSEMENT: Le tracking vidéo ne fonctionnera PAS avec transformers")
            print("⚠️  Pour la vidéo, installez le repo GitHub avec:")
            print("     ./install_sam3_github.sh")

            # Print memory stats after loading
            if self.enable_optimizations:
                print("\n[SAM3] Memory after loading:")
                stats_after = self.memory_manager.get_stats()
                print(stats_after)

            return

        except ImportError as e:
            print(f"[SAM3] ⚠️  Transformers SAM3 non disponible: {e}")
        except Exception as e:
            print(f"[SAM3] ⚠️  Échec transformers: {e}")
            traceback.print_exc()

        # Si on arrive ici, les deux méthodes ont échoué
        error_msg = f"""❌ Impossible de charger SAM3 avec aucune méthode!

🔧 SOLUTIONS:

1. REPO GITHUB (RECOMMANDÉ - Support vidéo complet):
   Utilisez le script d'installation fourni:
   ./install_sam3_github.sh

   OU manuellement:
   cd /tmp
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3
   pip install -e .

2. TRANSFORMERS (Simple - IMAGE SEULEMENT):
   pip install --upgrade transformers
   # Note: La vidéo ne fonctionnera PAS avec transformers

3. VÉRIFIER DÉPENDANCES:
   pip install pycocotools decord

Consultez QUICK_INSTALL.md pour plus de détails."""

        print(f"\n[SAM3 FATAL ERROR]\n{error_msg}", file=sys.stderr)
        raise RuntimeError(error_msg)

    def is_ready(self) -> bool:
        return self._image_processor is not None and self._video_predictor is not None

    @torch.no_grad()
    def segment_concept_image(self, image: Image.Image, text: str, threshold: float = 0.5, mask_threshold: float = 0.5) -> Dict[int, np.ndarray]:
        """Segmentation PCS (Promptable Concept Segmentation) sur une image avec un prompt texte."""
        if self._image_processor is None:
            raise RuntimeError("SAM3 backend not loaded")

        # Set image - returns state dict with backbone features
        state = self._image_processor.set_image(image, state=None)

        # Set confidence threshold
        state = self._image_processor.set_confidence_threshold(threshold, state=state)

        # Set text prompt - NOTE: prompt first, state second (correct API!)
        state = self._image_processor.set_text_prompt(prompt=text, state=state)

        # Extract masks - API returns torch tensors, not lists
        masks = state.get("masks", None)  # torch.Tensor (N, H, W)
        scores = state.get("scores", None)  # torch.Tensor (N,)
        boxes = state.get("boxes", None)  # torch.Tensor (N, 4)

        # Filter by threshold and create dict
        out: Dict[int, np.ndarray] = {}
        obj_id = 1

        if masks is not None and scores is not None:
            # Convert to numpy for iteration
            if hasattr(masks, 'cpu'):
                masks_np = masks.cpu().numpy() if masks.is_cuda else masks.numpy()
                scores_np = scores.cpu().numpy() if scores.is_cuda else scores.numpy()
            else:
                masks_np = np.array(masks)
                scores_np = np.array(scores)

            for i in range(len(masks_np)):
                if scores_np[i] >= threshold:
                    out[obj_id] = _mask_to_u8(masks_np[i])
                    obj_id += 1

        return out

    @torch.no_grad()
    def segment_interactive_image(self, image: Image.Image, points: List[Tuple[int,int,int]], boxes: List[Tuple[int,int,int,int,int]], multimask: bool = False) -> np.ndarray:
        """Segmentation PVS (Promptable Visual Segmentation) sur une image avec points/boxes.

        Note: SAM3 API uses add_geometric_prompt with normalized box coordinates.
        Points are NOT directly supported - must convert to bounding boxes.
        """
        if self._image_processor is None:
            raise RuntimeError("SAM3 backend not loaded")

        # Set image - returns state dict
        state = self._image_processor.set_image(image, state=None)

        # Get image dimensions for normalization
        w, h = image.size

        # Convert points to bounding box if no boxes provided
        if points and not boxes:
            # Calculate bounding box from points
            xs = [x for x, y, label in points if label == 1]  # positive points only
            ys = [y for x, y, label in points if label == 1]

            if xs and ys:
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

                # Add margin
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)

                boxes = [(x1, y1, x2, y2, 1)]

        # Process boxes
        if boxes:
            # Use first box with positive label
            for x1, y1, x2, y2, label in boxes:
                if label == 1:
                    # Convert to normalized center-based coordinates
                    # API expects: [center_x, center_y, width, height] normalized
                    center_x = (x1 + x2) / 2.0 / w
                    center_y = (y1 + y2) / 2.0 / h
                    box_width = (x2 - x1) / w
                    box_height = (y2 - y1) / h

                    # Add geometric prompt
                    state = self._image_processor.add_geometric_prompt(
                        box=[center_x, center_y, box_width, box_height],
                        label=True,  # positive prompt
                        state=state
                    )
                    break  # Use first valid box

        if not boxes:
            raise ValueError("No valid prompts provided")

        # Extract masks
        masks = state.get("masks", None)
        scores = state.get("scores", None)

        if masks is None or len(masks) == 0:
            # Return empty mask
            return np.zeros((h, w), dtype=np.uint8)

        # Take highest scoring mask
        if hasattr(masks, 'cpu'):
            masks_np = masks.cpu().numpy() if masks.is_cuda else masks.numpy()
            if scores is not None:
                scores_np = scores.cpu().numpy() if scores.is_cuda else scores.numpy()
                best_idx = np.argmax(scores_np)
            else:
                best_idx = 0
        else:
            masks_np = np.array(masks)
            best_idx = 0

        return _mask_to_u8(masks_np[best_idx])

    @torch.no_grad()
    def process_video_concept(self, frames: Sequence[Image.Image], texts: List[str],
                              threshold: float = 0.5) -> Iterator[FrameMasks]:
        """Segmentation PCS frame-par-frame (simplifié, plus robuste que tracking vidéo).

        Cette méthode utilise la segmentation IMAGE sur chaque frame au lieu du tracking vidéo.
        Avantages:
        - API simple et robuste
        - Fonctionne avec transformers ET GitHub SAM3
        - Meilleure qualité selon les reviewers
        - Pas de gestion de session complexe

        Le temporal smoothing est fait en post-processing par l'application.
        """
        logger.info(f"process_video_concept: début (frames={len(frames)}, texts={texts})")

        if self._image_processor is None:
            logger.error("process_video_concept: SAM3 backend not loaded")
            raise RuntimeError("SAM3 backend not loaded")

        if not texts:
            logger.error("process_video_concept: no text prompts provided")
            raise ValueError("At least one text prompt is required")

        total_frames = len(frames)
        logger.info(f"[SAM3 Video Simple] Processing {total_frames} frames with {len(texts)} prompts")
        print(f"[SAM3 Video Simple] Processing {total_frames} frames...")

        for frame_idx, frame in enumerate(frames):
            logger.debug(f"process_video_concept: processing frame {frame_idx}/{total_frames}")

            # Afficher progression tous les 10 frames
            if frame_idx % 10 == 0:
                print(f"[SAM3 Video Simple] Frame {frame_idx}/{total_frames}...")

            try:
                # Segmenter cette frame avec chaque prompt texte
                masks_by_id = {}
                obj_id = 1

                for text in texts:
                    logger.debug(f"process_video_concept: frame {frame_idx}, prompt '{text}'")

                    # Utiliser la segmentation IMAGE (simple et robuste)
                    frame_masks = self.segment_concept_image(
                        frame,
                        text=text,
                        threshold=threshold
                    )

                    # Ajouter les masks trouvés
                    for mask_obj_id, mask in frame_masks.items():
                        masks_by_id[obj_id] = mask
                        obj_id += 1

                # Yield les résultats pour cette frame
                if masks_by_id:
                    logger.debug(f"process_video_concept: frame {frame_idx} -> {len(masks_by_id)} masks")
                    yield FrameMasks(frame_idx=frame_idx, masks_by_id=masks_by_id)
                else:
                    logger.warning(f"process_video_concept: frame {frame_idx} -> no masks found")
                    # Yield frame vide pour maintenir la continuité
                    yield FrameMasks(frame_idx=frame_idx, masks_by_id={})

            except Exception as e:
                logger.error(f"process_video_concept: error on frame {frame_idx}: {e}", exc_info=True)
                print(f"[SAM3 Video Simple] Warning: Frame {frame_idx} failed: {e}")
                # Yield frame vide en cas d'erreur
                yield FrameMasks(frame_idx=frame_idx, masks_by_id={})

        logger.info(f"process_video_concept: completed {total_frames} frames")
        print(f"[SAM3 Video Simple] ✓ Completed {total_frames} frames")

    @torch.no_grad()
    def process_video_interactive(self, frames: Sequence[Image.Image],
                                  prompts: Dict[int, Dict[int, List[Tuple[int,int,int]]]]) -> Iterator[FrameMasks]:
        """Segmentation PVS frame-par-frame avec propagation de keyframes (simplifié).

        Utilise la segmentation IMAGE sur chaque frame. Pour les frames sans prompts,
        utilise les prompts de la frame clé précédente.
        """
        logger.info(f"process_video_interactive: début (frames={len(frames)}, keyframes={list(prompts.keys())})")

        if self._image_processor is None:
            logger.error("process_video_interactive: SAM3 backend not loaded")
            raise RuntimeError("SAM3 backend not loaded")

        if not prompts:
            logger.error("process_video_interactive: no prompts provided")
            raise ValueError("At least one prompt keyframe is required")

        total_frames = len(frames)
        logger.info(f"[SAM3 Video Interactive Simple] Processing {total_frames} frames with {len(prompts)} keyframes")
        print(f"[SAM3 Video Interactive Simple] Processing {total_frames} frames...")

        # Trouver la première frame avec prompts
        keyframe_indices = sorted(prompts.keys())
        if not keyframe_indices:
            raise ValueError("No keyframes with prompts")

        # Traiter chaque frame
        for frame_idx in range(total_frames):
            logger.debug(f"process_video_interactive: processing frame {frame_idx}/{total_frames}")

            if frame_idx % 10 == 0:
                print(f"[SAM3 Video Interactive Simple] Frame {frame_idx}/{total_frames}...")

            try:
                # Trouver la keyframe la plus proche (précédente)
                active_keyframe = None
                for kf in reversed(keyframe_indices):
                    if kf <= frame_idx:
                        active_keyframe = kf
                        break

                if active_keyframe is None:
                    # Avant la première keyframe, utiliser la première
                    active_keyframe = keyframe_indices[0]

                # Utiliser les prompts de cette keyframe
                frame_prompts = prompts[active_keyframe]

                masks_by_id = {}

                # Segmenter avec les prompts de chaque objet
                for obj_id, points in frame_prompts.items():
                    if not points:
                        continue

                    logger.debug(f"process_video_interactive: frame {frame_idx}, obj {obj_id}, {len(points)} points")

                    # Convertir les points au format attendu
                    boxes = []
                    pts = []
                    for x, y, label in points:
                        if label >= 0:  # Point positif/négatif
                            pts.append((x, y, label))
                        # On pourrait aussi gérer les boxes ici si besoin

                    if pts or boxes:
                        # Utiliser la segmentation IMAGE interactive
                        mask = self.segment_interactive_image(
                            frames[frame_idx],
                            points=pts,
                            boxes=boxes
                        )

                        if mask is not None and mask.max() > 0:
                            masks_by_id[obj_id] = mask

                # Yield les résultats
                if masks_by_id:
                    logger.debug(f"process_video_interactive: frame {frame_idx} -> {len(masks_by_id)} objects")
                    yield FrameMasks(frame_idx=frame_idx, masks_by_id=masks_by_id)
                else:
                    logger.warning(f"process_video_interactive: frame {frame_idx} -> no masks")
                    yield FrameMasks(frame_idx=frame_idx, masks_by_id={})

            except Exception as e:
                logger.error(f"process_video_interactive: error on frame {frame_idx}: {e}", exc_info=True)
                print(f"[SAM3 Video Interactive Simple] Warning: Frame {frame_idx} failed: {e}")
                yield FrameMasks(frame_idx=frame_idx, masks_by_id={})

        logger.info(f"process_video_interactive: completed {total_frames} frames")
        print(f"[SAM3 Video Interactive Simple] ✓ Completed {total_frames} frames")
