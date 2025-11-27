from __future__ import annotations
import numpy as np

def _rgb01(rgb_u8):
    return rgb_u8.astype(np.float32)/255.0

def estimate_bg_color(rgb_u8: np.ndarray, alpha_u8: np.ndarray, sample_max: int = 20000) -> np.ndarray:
    rgb = _rgb01(rgb_u8)
    a = alpha_u8.astype(np.float32)/255.0
    bg_mask = (a < 0.05).reshape(-1)
    pix = rgb.reshape(-1,3)[bg_mask]
    if pix.size == 0:
        return np.array([0.0,0.0,0.0], np.float32)
    if pix.shape[0] > sample_max:
        idx = np.random.choice(pix.shape[0], sample_max, replace=False)
        pix = pix[idx]
    return pix.mean(axis=0).astype(np.float32)

def despill_green_average(rgb_u8: np.ndarray, strength: float = 0.8) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    rgb = _rgb01(rgb_u8)
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    # clamp green towards avg of R/B
    target = (r + b) * 0.5
    g2 = g*(1.0-strength) + np.minimum(g, target) * strength
    out = np.stack([r,g2,b], axis=-1)
    return np.clip(out*255.0,0,255).astype(np.uint8)

def despill_blue_average(rgb_u8: np.ndarray, strength: float = 0.8) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    rgb = _rgb01(rgb_u8)
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    target = (r + g) * 0.5
    b2 = b*(1.0-strength) + np.minimum(b, target) * strength
    out = np.stack([r,g,b2], axis=-1)
    return np.clip(out*255.0,0,255).astype(np.uint8)

def luminance_restore(src_rgb_u8: np.ndarray, dst_rgb_u8: np.ndarray, amount: float = 1.0) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    src = _rgb01(src_rgb_u8)
    dst = _rgb01(dst_rgb_u8)
    # simple luma
    l_src = (0.2126*src[...,0] + 0.7152*src[...,1] + 0.0722*src[...,2])
    l_dst = (0.2126*dst[...,0] + 0.7152*dst[...,1] + 0.0722*dst[...,2])
    ratio = np.where(l_dst>1e-6, l_src/(l_dst+1e-6), 1.0)
    ratio = ratio[...,None]
    out = dst*(1.0-amount) + (dst*ratio)*amount
    return np.clip(out*255.0,0,255).astype(np.uint8)

def despill_physical(rgb_u8: np.ndarray, alpha_u8: np.ndarray, bg_rgb01: np.ndarray, edge_only: bool = True) -> np.ndarray:
    """Déspill 'unmix' simple : on retire une partie du channel spill (supposé green) près des bords.
    Ce n'est pas un modèle physique parfait, mais c'est une bonne base pour Flame : edge-only + luma restore.
    """
    rgb = _rgb01(rgb_u8)
    a = alpha_u8.astype(np.float32)/255.0
    # edge mask : zones semi-transparentes
    if edge_only:
        edge = np.clip(1.0 - np.abs(a*2.0-1.0), 0.0, 1.0)
    else:
        edge = 1.0 - a
    edge = edge[...,None]

    bg = bg_rgb01.reshape(1,1,3).astype(np.float32)
    # assume spill in G: remove green component that exceeds average of R/B + background hint
    r,g,b = rgb[...,0:1], rgb[...,1:2], rgb[...,2:3]
    target_g = (r+b)*0.5 + 0.25*bg[...,1:2]
    g2 = np.minimum(g, target_g)
    out = np.concatenate([r,g*(1-edge)+g2*edge,b], axis=-1)
    return np.clip(out*255.0,0,255).astype(np.uint8)
