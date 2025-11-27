from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Any
import numpy as np
import torch
from PIL import Image
import tempfile
import shutil
from pathlib import Path

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
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16 if self.device.type == "cuda" else torch.float32

        self.model_id = "facebook/sam3-hiera-large"
        self._image_model = None
        self._image_processor = None
        self._video_predictor = None

    def load(self, model_id_or_path: str = "facebook/sam3-hiera-large"):
        """Charge les modèles SAM3 pour image et vidéo."""
        import traceback
        import sys

        try:
            print(f"[SAM3] Début du chargement...")
            print(f"[SAM3] Device: {self.device}, dtype: {self.dtype}")

            try:
                from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
                from sam3.model.sam3_image_processor import Sam3Processor
                print("[SAM3] Imports SAM3 réussis")
            except ImportError as e:
                print(f"[SAM3 ERROR] Échec de l'import SAM3: {e}", file=sys.stderr)
                traceback.print_exc()
                raise RuntimeError(
                    "SAM3 n'est pas installé. Clone le repo officiel:\n"
                    "git clone https://github.com/facebookresearch/sam3.git\n"
                    "cd sam3\n"
                    "pip install -e .\n"
                    f"Import error: {e}"
                )

            self.model_id = model_id_or_path
            print(f"[SAM3] Model ID/Path: {model_id_or_path}")

            # Image model + processor
            print(f"[SAM3] Chargement SAM3 image model...")
            try:
                self._image_model = build_sam3_image_model(checkpoint=model_id_or_path)
                print(f"[SAM3] Image model construit, déplacement vers {self.device}...")
                self._image_model = self._image_model.to(self.device)
                if self.dtype in (torch.float16, torch.bfloat16):
                    print(f"[SAM3] Conversion en {self.dtype}...")
                    self._image_model = self._image_model.to(dtype=self.dtype)
                self._image_model.eval()
                print("[SAM3] Création du processor...")
                self._image_processor = Sam3Processor(self._image_model)
                print("[SAM3] ✅ Image model OK")
            except Exception as e:
                print(f"[SAM3 ERROR] Échec du chargement image model: {e}", file=sys.stderr)
                traceback.print_exc()
                raise

            # Video predictor
            print(f"[SAM3] Chargement SAM3 video predictor...")
            try:
                self._video_predictor = build_sam3_video_predictor(checkpoint=model_id_or_path)
                print(f"[SAM3] Video predictor construit, déplacement vers {self.device}...")
                self._video_predictor.model = self._video_predictor.model.to(self.device)
                if self.dtype in (torch.float16, torch.bfloat16):
                    print(f"[SAM3] Conversion en {self.dtype}...")
                    self._video_predictor.model = self._video_predictor.model.to(dtype=self.dtype)
                self._video_predictor.model.eval()
                print("[SAM3] ✅ Video predictor OK")
            except Exception as e:
                print(f"[SAM3 ERROR] Échec du chargement video predictor: {e}", file=sys.stderr)
                traceback.print_exc()
                raise

            print("✅ SAM3 chargé avec succès")

        except Exception as e:
            print(f"[SAM3 FATAL ERROR] Échec global du chargement: {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    def is_ready(self) -> bool:
        return self._image_processor is not None and self._video_predictor is not None

    @torch.no_grad()
    def segment_concept_image(self, image: Image.Image, text: str, threshold: float = 0.5, mask_threshold: float = 0.5) -> Dict[int, np.ndarray]:
        """Segmentation PCS (Promptable Concept Segmentation) sur une image avec un prompt texte."""
        if self._image_processor is None:
            raise RuntimeError("SAM3 backend not loaded")

        # Set image
        inference_state = self._image_processor.set_image(image)

        # Set text prompt
        output = self._image_processor.set_text_prompt(
            state=inference_state,
            prompt=text
        )

        # Extract masks
        masks = output.get("masks", [])  # List of masks
        scores = output.get("scores", [])

        # Filter by threshold and create dict
        out: Dict[int, np.ndarray] = {}
        obj_id = 1

        if isinstance(masks, list):
            for i, (mask, score) in enumerate(zip(masks, scores)):
                if score >= threshold:
                    out[obj_id] = _mask_to_u8(mask)
                    obj_id += 1
        elif hasattr(masks, 'shape'):  # tensor
            for i in range(masks.shape[0]):
                if scores[i] >= threshold:
                    out[obj_id] = _mask_to_u8(masks[i])
                    obj_id += 1

        return out

    @torch.no_grad()
    def segment_interactive_image(self, image: Image.Image, points: List[Tuple[int,int,int]], boxes: List[Tuple[int,int,int,int,int]], multimask: bool = False) -> np.ndarray:
        """Segmentation PVS (Promptable Visual Segmentation) sur une image avec points/boxes."""
        if self._image_processor is None:
            raise RuntimeError("SAM3 backend not loaded")

        # Set image
        inference_state = self._image_processor.set_image(image)

        # Prepare prompts
        prompt_points = []
        prompt_labels = []
        prompt_boxes = []

        if points:
            for x, y, label in points:
                prompt_points.append([int(x), int(y)])
                prompt_labels.append(int(label))

        if boxes:
            # Use last box with positive label
            for x1, y1, x2, y2, label in boxes:
                if label == 1:
                    prompt_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        # Set prompts
        if prompt_points:
            output = self._image_processor.set_point_prompt(
                state=inference_state,
                points=np.array(prompt_points),
                labels=np.array(prompt_labels)
            )
        elif prompt_boxes:
            output = self._image_processor.set_box_prompt(
                state=inference_state,
                box=np.array(prompt_boxes[0])  # Use first box
            )
        else:
            raise ValueError("No prompts provided")

        # Extract best mask
        masks = output.get("masks", [])
        if len(masks) == 0:
            # Return empty mask
            w, h = image.size
            return np.zeros((h, w), dtype=np.uint8)

        # Take first/best mask
        best_mask = masks[0] if isinstance(masks, list) else masks[0]
        return _mask_to_u8(best_mask)

    @torch.no_grad()
    def track_concept_video(self, frames: Sequence[Image.Image], texts: List[str]) -> Iterator[FrameMasks]:
        """Tracking PCS vidéo avec prompts texte."""
        if self._video_predictor is None:
            raise RuntimeError("SAM3 backend not loaded")

        # Save frames to temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="sam3_video_"))
        try:
            # Save frames as JPEG sequence
            for i, frame in enumerate(frames):
                frame.convert("RGB").save(temp_dir / f"{i:06d}.jpg", quality=95)

            # Start session
            response = self._video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=str(temp_dir),
                )
            )
            session_id = response["session_id"]

            # Add text prompts on first frame
            for text in texts:
                response = self._video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        text=text,
                    )
                )

            # Propagate through video
            response = self._video_predictor.handle_request(
                request=dict(
                    type="propagate",
                    session_id=session_id,
                )
            )

            # Extract results per frame
            outputs = response.get("outputs", {})
            for frame_idx in sorted(outputs.keys()):
                frame_output = outputs[frame_idx]
                masks = frame_output.get("masks", [])
                object_ids = frame_output.get("object_ids", list(range(1, len(masks) + 1)))

                masks_by_id = {}
                for obj_id, mask in zip(object_ids, masks):
                    masks_by_id[int(obj_id)] = _mask_to_u8(mask)

                yield FrameMasks(frame_idx=int(frame_idx), masks_by_id=masks_by_id)

            # End session
            self._video_predictor.handle_request(
                request=dict(
                    type="end_session",
                    session_id=session_id,
                )
            )

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    @torch.no_grad()
    def track_interactive_video(self, frames: Sequence[Image.Image], prompts: Dict[int, Dict[int, List[Tuple[int,int,int]]]]) -> Iterator[FrameMasks]:
        """Tracking PVS vidéo avec keyframes interactifs."""
        if self._video_predictor is None:
            raise RuntimeError("SAM3 backend not loaded")

        # Save frames to temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="sam3_video_"))
        try:
            # Save frames as JPEG sequence
            for i, frame in enumerate(frames):
                frame.convert("RGB").save(temp_dir / f"{i:06d}.jpg", quality=95)

            # Start session
            response = self._video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=str(temp_dir),
                )
            )
            session_id = response["session_id"]

            # Add point prompts on keyframes
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

            # Propagate through video
            response = self._video_predictor.handle_request(
                request=dict(
                    type="propagate",
                    session_id=session_id,
                )
            )

            # Extract results per frame
            outputs = response.get("outputs", {})
            for frame_idx in sorted(outputs.keys()):
                frame_output = outputs[frame_idx]
                masks = frame_output.get("masks", [])
                object_ids = frame_output.get("object_ids", list(range(1, len(masks) + 1)))

                masks_by_id = {}
                for obj_id, mask in zip(object_ids, masks):
                    masks_by_id[int(obj_id)] = _mask_to_u8(mask)

                yield FrameMasks(frame_idx=int(frame_idx), masks_by_id=masks_by_id)

            # End session
            self._video_predictor.handle_request(
                request=dict(
                    type="end_session",
                    session_id=session_id,
                )
            )

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
