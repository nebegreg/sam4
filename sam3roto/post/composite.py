from __future__ import annotations
import numpy as np

def premultiply(rgb_u8: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32)/255.0
    a = (alpha_u8.astype(np.float32)/255.0)[...,None]
    out = rgb*a
    return np.clip(out*255.0,0,255).astype(np.uint8)
