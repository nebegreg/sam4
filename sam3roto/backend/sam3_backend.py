from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Any
import numpy as np
import torch
from PIL import Image
import tempfile
import shutil
from pathlib import Path

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

        # MÉTHODE 1: Essayer transformers (HuggingFace) - VOTRE CODE QUI MARCHAIT
        try:
            print("\n[SAM3] 🔄 Tentative 1: Transformers/HuggingFace...")
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

            # Stocker dans les attributs (interface simplifiée pour transformers)
            self._image_model = model
            self._image_processor = processor
            self._video_predictor = model  # Utiliser le même modèle pour vidéo
            self._use_transformers = True

            print("✅ SAM3 chargé avec succès (transformers)")

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

        # MÉTHODE 2: Essayer repo GitHub officiel
        try:
            print("\n[SAM3] 🔄 Tentative 2: Repo GitHub officiel...")
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
            self._image_model = build_sam3_image_model(
                checkpoint_path=model_id_or_path if not load_from_hf else None,
                load_from_HF=load_from_hf,
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
            self._video_predictor = build_sam3_video_predictor(
                checkpoint_path=model_id_or_path if not load_from_hf else None,
                load_from_HF=load_from_hf,
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
            print(f"[SAM3] ❌ Repo GitHub non disponible: {e}", file=sys.stderr)
            traceback.print_exc()
        except Exception as e:
            print(f"[SAM3] ❌ Échec repo GitHub: {e}", file=sys.stderr)
            traceback.print_exc()

        # Si on arrive ici, les deux méthodes ont échoué
        error_msg = f"""❌ Impossible de charger SAM3 avec aucune méthode!

🔧 SOLUTIONS:

1. TRANSFORMERS (Simple):
   pip install --upgrade transformers
   # Puis utiliser: 'facebook/sam3-hiera-large'

2. REPO GITHUB (Complet):
   cd ~/Documents/venv_sam/.external_models
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3
   pip install -e .
   # Puis utiliser le chemin local du modèle

3. VÉRIFIER DÉPENDANCES:
   pip install pycocotools decord

Consultez INSTALLATION_RAPIDE.md pour plus de détails."""

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
    def track_concept_video(self, frames: Sequence[Image.Image], texts: List[str]) -> Iterator[FrameMasks]:
        """Tracking PCS vidéo avec prompts texte."""
        if self._video_predictor is None:
            raise RuntimeError("SAM3 backend not loaded")

        if not texts:
            raise ValueError("At least one text prompt is required")

        # Save frames to temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="sam3_video_"))
        session_id = None

        try:
            # Save frames as JPEG sequence
            print(f"[SAM3 Video] Saving {len(frames)} frames to temp dir...")
            for i, frame in enumerate(frames):
                frame.convert("RGB").save(temp_dir / f"{i:06d}.jpg", quality=95)

            # Start session
            print("[SAM3 Video] Starting video session...")
            response = self._video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=str(temp_dir),
                )
            )

            if "session_id" not in response:
                raise RuntimeError(f"Failed to start session: {response}")

            session_id = response["session_id"]
            print(f"[SAM3 Video] Session started: {session_id}")

            # Add text prompts on first frame
            print(f"[SAM3 Video] Adding {len(texts)} text prompts...")
            for i, text in enumerate(texts):
                response = self._video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        text=text,
                    )
                )

                if "error" in response:
                    print(f"[SAM3 Video] Warning: Prompt {i} failed: {response['error']}")

            # Propagate through video
            print("[SAM3 Video] Propagating through video...")
            response = self._video_predictor.handle_request(
                request=dict(
                    type="propagate",
                    session_id=session_id,
                )
            )

            if "error" in response:
                raise RuntimeError(f"Propagation failed: {response['error']}")

            # Extract results per frame
            outputs = response.get("outputs", {})
            if not outputs:
                print("[SAM3 Video] Warning: No outputs from propagation")
                return

            print(f"[SAM3 Video] Got results for {len(outputs)} frames")

            for frame_idx in sorted(outputs.keys()):
                frame_output = outputs[frame_idx]
                masks = frame_output.get("masks", [])
                object_ids = frame_output.get("object_ids", list(range(1, len(masks) + 1)))

                masks_by_id = {}
                for obj_id, mask in zip(object_ids, masks):
                    masks_by_id[int(obj_id)] = _mask_to_u8(mask)

                yield FrameMasks(frame_idx=int(frame_idx), masks_by_id=masks_by_id)

        except Exception as e:
            print(f"[SAM3 Video] Error during tracking: {e}")
            raise

        finally:
            # End session if it was created
            if session_id is not None:
                try:
                    print(f"[SAM3 Video] Ending session {session_id}...")
                    self._video_predictor.handle_request(
                        request=dict(
                            type="end_session",
                            session_id=session_id,
                        )
                    )
                except Exception as e:
                    print(f"[SAM3 Video] Warning: Failed to end session: {e}")

            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=False)
                print("[SAM3 Video] Temp directory cleaned up")
            except Exception as e:
                print(f"[SAM3 Video] Warning: Failed to cleanup temp dir: {e}")

    @torch.no_grad()
    def track_interactive_video(self, frames: Sequence[Image.Image], prompts: Dict[int, Dict[int, List[Tuple[int,int,int]]]]) -> Iterator[FrameMasks]:
        """Tracking PVS vidéo avec keyframes interactifs."""
        if self._video_predictor is None:
            raise RuntimeError("SAM3 backend not loaded")

        if not prompts:
            raise ValueError("At least one prompt keyframe is required")

        # Save frames to temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="sam3_video_"))
        session_id = None

        try:
            # Save frames as JPEG sequence
            print(f"[SAM3 Video Interactive] Saving {len(frames)} frames...")
            for i, frame in enumerate(frames):
                frame.convert("RGB").save(temp_dir / f"{i:06d}.jpg", quality=95)

            # Start session
            print("[SAM3 Video Interactive] Starting video session...")
            response = self._video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=str(temp_dir),
                )
            )

            if "session_id" not in response:
                raise RuntimeError(f"Failed to start session: {response}")

            session_id = response["session_id"]
            print(f"[SAM3 Video Interactive] Session started: {session_id}")

            # Add point prompts on keyframes
            print(f"[SAM3 Video Interactive] Adding prompts on {len(prompts)} keyframes...")
            for frame_idx, objs in prompts.items():
                for obj_id, points in objs.items():
                    prompt_points = [[int(x), int(y)] for x, y, _ in points]
                    prompt_labels = [int(label) for _, _, label in points]

                    response = self._video_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=int(frame_idx),
                            points=prompt_points,
                            labels=prompt_labels,
                            object_id=int(obj_id),
                        )
                    )

                    if "error" in response:
                        print(f"[SAM3 Video Interactive] Warning: Prompt on frame {frame_idx}, obj {obj_id} failed: {response['error']}")

            # Propagate through video
            print("[SAM3 Video Interactive] Propagating through video...")
            response = self._video_predictor.handle_request(
                request=dict(
                    type="propagate",
                    session_id=session_id,
                )
            )

            if "error" in response:
                raise RuntimeError(f"Propagation failed: {response['error']}")

            # Extract results per frame
            outputs = response.get("outputs", {})
            if not outputs:
                print("[SAM3 Video Interactive] Warning: No outputs from propagation")
                return

            print(f"[SAM3 Video Interactive] Got results for {len(outputs)} frames")

            for frame_idx in sorted(outputs.keys()):
                frame_output = outputs[frame_idx]
                masks = frame_output.get("masks", [])
                object_ids = frame_output.get("object_ids", list(range(1, len(masks) + 1)))

                masks_by_id = {}
                for obj_id, mask in zip(object_ids, masks):
                    masks_by_id[int(obj_id)] = _mask_to_u8(mask)

                yield FrameMasks(frame_idx=int(frame_idx), masks_by_id=masks_by_id)

        except Exception as e:
            print(f"[SAM3 Video Interactive] Error during tracking: {e}")
            raise

        finally:
            # End session if it was created
            if session_id is not None:
                try:
                    print(f"[SAM3 Video Interactive] Ending session {session_id}...")
                    self._video_predictor.handle_request(
                        request=dict(
                            type="end_session",
                            session_id=session_id,
                        )
                    )
                except Exception as e:
                    print(f"[SAM3 Video Interactive] Warning: Failed to end session: {e}")

            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=False)
                print("[SAM3 Video Interactive] Temp directory cleaned up")
            except Exception as e:
                print(f"[SAM3 Video Interactive] Warning: Failed to cleanup temp dir: {e}")
