from __future__ import annotations
import numpy as np
import cv2

def _ensure_u8(m):
    return np.asarray(m, dtype=np.uint8)

def fill_small_holes(mask_u8: np.ndarray, max_area: int) -> np.ndarray:
    if max_area <= 0:
        return _ensure_u8(mask_u8)
    m = _ensure_u8(mask_u8)
    h, w = m.shape[:2]
    # inverse mask and find connected comps inside foreground region
    inv = cv2.bitwise_not(m)
    # holes are components that are not touching border in inv where original fg exists around
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = m.copy()
    for i in range(1, num):
        x,y,ww,hh,area = stats[i]
        if area <= max_area:
            # check if touches border: if it does, it's background, not a hole
            if x == 0 or y == 0 or x+ww >= w or y+hh >= h:
                continue
            out[labels == i] = 255
    return out

def remove_small_dots(mask_u8: np.ndarray, max_area: int) -> np.ndarray:
    if max_area <= 0:
        return _ensure_u8(mask_u8)
    m = _ensure_u8(mask_u8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((m>0).astype(np.uint8), connectivity=8)
    out = m.copy()
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= max_area:
            out[labels == i] = 0
    return out

def grow_shrink(mask_u8: np.ndarray, amount: int) -> np.ndarray:
    m = _ensure_u8(mask_u8)
    if amount == 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(amount)*2+1, abs(amount)*2+1))
    if amount > 0:
        return cv2.dilate(m, k, iterations=1)
    else:
        return cv2.erode(m, k, iterations=1)

def border_fix(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    m = _ensure_u8(mask_u8)
    if radius <= 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

def feather_alpha(mask_u8: np.ndarray, radius: float) -> np.ndarray:
    m = _ensure_u8(mask_u8)
    if radius <= 0.0:
        return (m>0).astype(np.uint8)*255
    # smooth edges with gaussian blur
    k = int(max(3, (radius*2)//1 * 2 + 1))
    blur = cv2.GaussianBlur(m.astype(np.float32)/255.0, (k,k), sigmaX=radius, sigmaY=radius)
    return np.clip(blur*255.0, 0, 255).astype(np.uint8)

def alpha_from_trimap(mask_u8: np.ndarray, band: int) -> np.ndarray:
    """Crée un alpha doux basé sur distance transform.
    band = largeur de la zone inconnue autour du bord."""
    m = (mask_u8>0).astype(np.uint8)
    if band <= 1:
        return m.astype(np.uint8)*255

    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(1-m, cv2.DIST_L2, 3)

    # alpha 1 à l'intérieur, 0 à l'extérieur, transition dans un band autour du bord
    a = np.clip((dist_in - dist_out + band) / (2.0*band), 0.0, 1.0)
    # léger débruitage sur alpha
    a = cv2.GaussianBlur(a.astype(np.float32), (0,0), 0.75)
    return np.clip(a*255.0, 0, 255).astype(np.uint8)

def temporal_smooth(prev_u8: np.ndarray, cur_u8: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    p = prev_u8.astype(np.float32)
    c = cur_u8.astype(np.float32)
    out = p*(1.0-strength) + c*strength
    return np.clip(out, 0, 255).astype(np.uint8)
