from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import subprocess

def save_png_gray(path: Path, gray_u8: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), gray_u8.astype(np.uint8))

def save_png_rgb(path: Path, rgb_u8: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb_u8.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)

def save_png_rgba(path: Path, rgba_u8: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV expects BGRA
    bgra = rgba_u8[..., [2,1,0,3]].astype(np.uint8)
    cv2.imwrite(str(path), bgra)

def try_export_prores4444_from_png_sequence(pattern: str, out_mov: Path, fps: float) -> bool:
    out_mov = Path(out_mov)
    out_mov.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg","-y",
        "-framerate", str(float(fps)),
        "-i", pattern,
        "-c:v","prores_ks",
        "-profile:v","4",   # 4444
        "-pix_fmt","yuva444p10le",
        str(out_mov),
    ]
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False
