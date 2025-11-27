"""
Module de matting avancé pour les détails fins (cheveux, fourrure, etc.)
Basé sur les techniques state-of-the-art :
- Guided Filter (He et al.) pour le raffinement d'alpha
- Information Flow matting (OpenCV)
- Trimap automatique avancé
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import Optional, Tuple

def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-4) -> np.ndarray:
    """
    Guided Filter pour le raffinement d'alpha matting.

    Référence: "Guided Image Filtering" - He et al. ECCV 2010
    http://kaiminghe.com/eccv10/

    Args:
        guide: Image guide RGB (H,W,3) float32 [0,1]
        src: Source alpha (H,W) float32 [0,1]
        radius: Rayon du filtre
        eps: Regularization parameter (plus petit = suit mieux l'image guide)

    Returns:
        Alpha raffiné (H,W) float32 [0,1]
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)

    if guide.ndim == 3:
        # Guide RGB: appliquer le filtre sur chaque canal
        h, w = src.shape[:2]
        result = np.zeros_like(src)

        for c in range(3):
            guide_c = guide[:, :, c]

            # Moyennes locales
            mean_I = cv2.boxFilter(guide_c, -1, (radius, radius))
            mean_p = cv2.boxFilter(src, -1, (radius, radius))
            mean_Ip = cv2.boxFilter(guide_c * src, -1, (radius, radius))

            # Covariance et variance
            cov_Ip = mean_Ip - mean_I * mean_p
            var_I = cv2.boxFilter(guide_c * guide_c, -1, (radius, radius)) - mean_I * mean_I

            # Coefficients linéaires
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I

            # Moyennes des coefficients
            mean_a = cv2.boxFilter(a, -1, (radius, radius))
            mean_b = cv2.boxFilter(b, -1, (radius, radius))

            # Output
            result += mean_a * guide_c

        result = result / 3.0 + mean_b
    else:
        # Guide grayscale
        mean_I = cv2.boxFilter(guide, -1, (radius, radius))
        mean_p = cv2.boxFilter(src, -1, (radius, radius))
        mean_Ip = cv2.boxFilter(guide * src, -1, (radius, radius))

        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = cv2.boxFilter(guide * guide, -1, (radius, radius)) - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))

        result = mean_a * guide + mean_b

    return np.clip(result, 0, 1)


def auto_trimap_advanced(mask_u8: np.ndarray, fg_threshold: int = 240, bg_threshold: int = 15,
                         erosion: int = 5, dilation: int = 10) -> np.ndarray:
    """
    Génération automatique d'un trimap avancé pour les détails fins.

    Référence: "Automatic Trimap Generation for Image Matting" - Liu et al. 2017
    https://arxiv.org/abs/1707.00333

    Args:
        mask_u8: Masque binaire (H,W) uint8 [0,255]
        fg_threshold: Seuil pour foreground certain (>= threshold)
        bg_threshold: Seuil pour background certain (<= threshold)
        erosion: Taille de l'érosion pour foreground certain
        dilation: Taille de la dilatation pour unknown region

    Returns:
        Trimap (H,W) uint8 où 0=BG, 128=Unknown, 255=FG
    """
    mask = mask_u8.astype(np.uint8)
    h, w = mask.shape

    # Foreground certain (érosion du masque)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion*2+1, erosion*2+1))
    fg_certain = cv2.erode((mask >= fg_threshold).astype(np.uint8), kernel_erode)

    # Background certain (inverse du masque dilaté)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation*2+1, dilation*2+1))
    fg_dilated = cv2.dilate((mask > bg_threshold).astype(np.uint8), kernel_dilate)
    bg_certain = 1 - fg_dilated

    # Unknown region
    trimap = np.zeros((h, w), dtype=np.uint8)
    trimap[bg_certain == 1] = 0      # Background
    trimap[fg_certain == 1] = 255    # Foreground
    trimap[(bg_certain == 0) & (fg_certain == 0)] = 128  # Unknown

    return trimap


def refine_alpha_for_hair(rgb_u8: np.ndarray, alpha_u8: np.ndarray,
                          mode: str = "guided", **kwargs) -> np.ndarray:
    """
    Raffine l'alpha spécifiquement pour les cheveux et détails fins.

    Args:
        rgb_u8: Image RGB (H,W,3) uint8 [0,255]
        alpha_u8: Alpha initial (H,W) uint8 [0,255]
        mode: "guided", "trimap", "both"
        kwargs: Paramètres additionnels (radius, eps, etc.)

    Returns:
        Alpha raffiné (H,W) uint8 [0,255]
    """
    rgb_f32 = rgb_u8.astype(np.float32) / 255.0
    alpha_f32 = alpha_u8.astype(np.float32) / 255.0

    if mode == "guided" or mode == "both":
        # Guided filter avec paramètres optimisés pour les cheveux
        radius = kwargs.get("radius", 8)
        eps = kwargs.get("eps", 1e-5)
        alpha_f32 = guided_filter(rgb_f32, alpha_f32, radius=radius, eps=eps)

    if mode == "trimap" or mode == "both":
        # Génération de trimap et raffinement
        trimap = auto_trimap_advanced(
            (alpha_f32 * 255).astype(np.uint8),
            fg_threshold=kwargs.get("fg_threshold", 240),
            bg_threshold=kwargs.get("bg_threshold", 15),
            erosion=kwargs.get("erosion", 3),
            dilation=kwargs.get("dilation", 8)
        )

        # Si OpenCV contrib est disponible, utiliser information flow matting
        try:
            # Information Flow Matting (très bon pour les cheveux)
            alpha_f32_255 = (alpha_f32 * 255).astype(np.uint8)
            refined = cv2.alphamat.infoFlow(rgb_u8, trimap)
            if refined is not None:
                alpha_f32 = refined.astype(np.float32) / 255.0
        except (AttributeError, cv2.error):
            # Fallback: utiliser guided filter avec le trimap comme guide
            trimap_f32 = trimap.astype(np.float32) / 255.0
            alpha_f32 = guided_filter(rgb_f32, trimap_f32, radius=10, eps=1e-6)

    # Clamp et convertir
    alpha_refined = np.clip(alpha_f32 * 255.0, 0, 255).astype(np.uint8)

    return alpha_refined


def edge_aware_smoothing(alpha_u8: np.ndarray, rgb_u8: np.ndarray,
                        sigma_space: float = 10.0, sigma_color: float = 25.0) -> np.ndarray:
    """
    Lissage edge-aware pour préserver les détails fins comme les cheveux.
    Utilise un filtre bilatéral guidé par l'image RGB.

    Args:
        alpha_u8: Alpha (H,W) uint8 [0,255]
        rgb_u8: Image RGB guide (H,W,3) uint8 [0,255]
        sigma_space: Paramètre spatial du filtre bilatéral
        sigma_color: Paramètre de couleur du filtre bilatéral

    Returns:
        Alpha lissé (H,W) uint8 [0,255]
    """
    # Joint bilateral filter: lisse alpha en suivant les edges de RGB
    d = int(sigma_space * 2)
    smoothed = cv2.ximgproc.jointBilateralFilter(
        rgb_u8, alpha_u8, d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    ) if hasattr(cv2, 'ximgproc') else cv2.bilateralFilter(alpha_u8, d, sigma_color, sigma_space)

    return smoothed.astype(np.uint8)


def detail_preserving_blur(alpha_u8: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Blur qui préserve les détails (cheveux fins, etc.) en utilisant un filtre de domaine.

    Args:
        alpha_u8: Alpha (H,W) uint8 [0,255]
        strength: Force du blur (0-1)

    Returns:
        Alpha avec détails préservés (H,W) uint8 [0,255]
    """
    if strength <= 0.0:
        return alpha_u8

    # Utiliser domain transform filter si disponible
    try:
        if hasattr(cv2, 'ximgproc'):
            dtf = cv2.ximgproc.createDomainTransformFilter(
                guide=alpha_u8,
                sigmaSpatial=20.0 * strength,
                sigmaColor=50.0 * strength
            )
            filtered = dtf.filter(alpha_u8)
            return filtered.astype(np.uint8)
    except:
        pass

    # Fallback: guided filter auto-guidé
    alpha_f32 = alpha_u8.astype(np.float32) / 255.0
    radius = int(5 * strength)
    eps = 0.01 * (1.0 - strength)

    # Mean filter
    mean = cv2.boxFilter(alpha_f32, -1, (radius*2+1, radius*2+1))
    var = cv2.boxFilter(alpha_f32 * alpha_f32, -1, (radius*2+1, radius*2+1)) - mean * mean

    a = var / (var + eps)
    b = (1 - a) * mean

    result = a * alpha_f32 + b
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def multi_scale_refinement(rgb_u8: np.ndarray, alpha_u8: np.ndarray,
                           scales: list = [1.0, 0.5, 0.25]) -> np.ndarray:
    """
    Raffinement multi-échelle pour capturer à la fois les structures globales
    et les détails fins comme les cheveux.

    Args:
        rgb_u8: Image RGB (H,W,3) uint8 [0,255]
        alpha_u8: Alpha initial (H,W) uint8 [0,255]
        scales: Liste des échelles à utiliser

    Returns:
        Alpha raffiné (H,W) uint8 [0,255]
    """
    h, w = alpha_u8.shape
    results = []

    for scale in scales:
        if scale != 1.0:
            # Downscale
            new_h, new_w = int(h * scale), int(w * scale)
            rgb_scaled = cv2.resize(rgb_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            alpha_scaled = cv2.resize(alpha_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            rgb_scaled = rgb_u8
            alpha_scaled = alpha_u8

        # Refine at this scale
        refined = refine_alpha_for_hair(rgb_scaled, alpha_scaled, mode="guided", radius=8, eps=1e-5)

        # Upscale back
        if scale != 1.0:
            refined = cv2.resize(refined, (w, h), interpolation=cv2.INTER_LINEAR)

        results.append(refined.astype(np.float32))

    # Combine results (weighted average, plus de poids sur les petites échelles pour les détails)
    weights = [0.5, 0.3, 0.2][:len(results)]
    weights = np.array(weights) / sum(weights)

    final = sum(r * w for r, w in zip(results, weights))
    return np.clip(final, 0, 255).astype(np.uint8)
