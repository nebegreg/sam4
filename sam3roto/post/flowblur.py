from __future__ import annotations
import numpy as np
import cv2

def edge_motion_blur_alpha(prev_rgb_u8: np.ndarray, cur_rgb_u8: np.ndarray, alpha_u8: np.ndarray, strength: float = 0.5, samples: int = 6) -> np.ndarray:
    """Approx motion blur pour l'alpha à partir d'optical flow (Farneback).
    - strength: 0..1
    - samples: nombre d'échantillons le long du vecteur de flow
    Cible : bords (alpha 0..255). Utilisation VFX : à utiliser léger.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0 or samples <= 1:
        return alpha_u8

    prev_g = cv2.cvtColor(prev_rgb_u8, cv2.COLOR_RGB2GRAY)
    cur_g = cv2.cvtColor(cur_rgb_u8, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_g, cur_g, None, 0.5, 3, 21, 3, 5, 1.2, 0)

    a = alpha_u8.astype(np.float32)/255.0
    # only around edges
    edge = (a > 0.02) & (a < 0.98)
    if not np.any(edge):
        return alpha_u8

    H, W = a.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    fx = flow[...,0]
    fy = flow[...,1]

    acc = a.copy()
    count = np.ones_like(a)
    for i in range(1, samples):
        t = (i / (samples-1)) * strength
        x2 = np.clip(xs - fx*t, 0, W-1)
        y2 = np.clip(ys - fy*t, 0, H-1)
        samp = cv2.remap(a, x2, y2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        acc[edge] += samp[edge]
        count[edge] += 1.0
    out = acc / np.maximum(count, 1e-6)
    return np.clip(out*255.0,0,255).astype(np.uint8)
