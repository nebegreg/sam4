from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import os
import re
import numpy as np
from PIL import Image

@dataclass
class MediaSource:
    path: str
    frames: List[Image.Image]
    fps: float

def _numeric_sort_key(p: str):
    # sort by last number in filename
    m = re.findall(r"(\d+)", os.path.basename(p))
    return (int(m[-1]) if m else 0, p)

def load_image_sequence(folder: str, fps: float = 25.0) -> MediaSource:
    folder = str(folder)
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".bmp")
    files = [str(p) for p in Path(folder).glob("*") if p.suffix.lower() in exts]
    files.sort(key=_numeric_sort_key)
    if not files:
        raise FileNotFoundError(f"Aucune image trouvée dans: {folder}")
    frames = [Image.open(f).convert("RGB") for f in files]
    return MediaSource(path=folder, frames=frames, fps=float(fps))

def load_video(path: str, max_frames: Optional[int] = None) -> MediaSource:
    # Prefer imageio (ffmpeg) then fallback to OpenCV
    path = str(path)

    frames: List[Image.Image] = []
    fps = 25.0
    try:
        import imageio.v3 as iio
        meta = iio.immeta(path)
        fps = float(meta.get("fps", 25.0) or 25.0)
        idx = 0
        for frame in iio.imiter(path):
            if max_frames is not None and idx >= max_frames:
                break
            # frame is HxWxC (RGB)
            img = Image.fromarray(frame[..., :3].astype(np.uint8), mode="RGB")
            frames.append(img)
            idx += 1
        if frames:
            return MediaSource(path=path, frames=frames, fps=fps)
    except Exception:
        pass

    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if max_frames is not None and idx >= max_frames:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError("Vidéo lue mais aucune frame décodée (codec?)")
    return MediaSource(path=path, frames=frames, fps=float(fps))
