from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import shutil
import numpy as np
from PIL import Image
import torch

@dataclass
class DA3Prediction:
    indices: List[int]
    depth: np.ndarray                 # [N,H,W]
    conf: Optional[np.ndarray]        # [N,H,W] or None
    extrinsics: Optional[np.ndarray]  # [N,3,4] or None
    intrinsics: Optional[np.ndarray]  # [N,3,3] or None

class DepthAnything3Backend:
    def __init__(self, tmp_root: Path):
        self.tmp_root = Path(tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.device = None
        self.model_id = ""

    def is_ready(self) -> bool:
        return self.model is not None

    def load(self, model_id: str = "depth-anything/DA3-BASE", device: Optional[str] = None) -> None:
        try:
            from depth_anything_3.api import DepthAnything3  # official DA3 API citeturn1view0
        except Exception as e:
            raise RuntimeError(
                "Depth Anything 3 n'est pas installé. Installe-le depuis GitHub (voir README). "
                f"Import error: {e}"
            )
        self.model_id = model_id
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DepthAnything3.from_pretrained(model_id).to(device=self.device)

    def _frames_to_dir(self, frames: Sequence[Image.Image], indices: List[int]) -> Tuple[Path, List[str]]:
        tmp_dir = self.tmp_root / "session"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        paths: List[str] = []
        for i in indices:
            img = frames[i].convert("RGB")
            fp = tmp_dir / f"{i:05d}.png"
            img.save(fp)
            paths.append(str(fp))
        return tmp_dir, paths

    @torch.no_grad()
    def infer(self, frames: Sequence[Image.Image], indices: Optional[Sequence[int]] = None) -> DA3Prediction:
        if self.model is None:
            raise RuntimeError("DA3 non chargé. Clique 'Charger DA3' d'abord.")
        if indices is None:
            indices = list(range(len(frames)))
        indices = list(indices)
        if not indices:
            raise ValueError("indices vide")
        tmp_dir, paths = self._frames_to_dir(frames, indices)
        try:
            pred = self.model.inference(paths)  # official API citeturn1view0
            depth = np.asarray(pred.depth, np.float32)
            conf = np.asarray(pred.conf, np.float32) if getattr(pred, "conf", None) is not None else None
            extr = np.asarray(pred.extrinsics, np.float32) if getattr(pred, "extrinsics", None) is not None else None
            intr = np.asarray(pred.intrinsics, np.float32) if getattr(pred, "intrinsics", None) is not None else None
            return DA3Prediction(indices=indices, depth=depth, conf=conf, extrinsics=extr, intrinsics=intr)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
