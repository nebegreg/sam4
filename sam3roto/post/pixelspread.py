from __future__ import annotations
import numpy as np
import cv2

def pixel_spread_rgb(rgb_u8: np.ndarray, alpha_u8: np.ndarray, radius: float = 8.0) -> np.ndarray:
    if radius <= 0:
        return rgb_u8
    rgb = rgb_u8.astype(np.uint8)
    a = alpha_u8.astype(np.uint8)
    # build a binary fg
    fg = (a > 0).astype(np.uint8) * 255
    k = int(max(1, round(radius)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k*2+1, k*2+1))
    dil = cv2.dilate(fg, kernel, iterations=1)
    edge = cv2.subtract(dil, fg)  # ring outside fg
    # fill ring with nearest fg color using inpaint on background region
    # Inpaint wants 8-bit 1-channel mask where non-zero pixels are inpainted
    mask = edge
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # inpaint uses neighbors to fill masked region
    filled = cv2.inpaint(bgr, mask, inpaintRadius=max(1, k//2), flags=cv2.INPAINT_TELEA)
    out = bgr.copy()
    out[mask>0] = filled[mask>0]
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
