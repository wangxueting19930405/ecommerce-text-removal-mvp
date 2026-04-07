from __future__ import annotations

import cv2
import numpy as np


def surrounding_ring(mask: np.ndarray, pad: int = 12) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    outer = cv2.dilate(mask, kernel, iterations=1)
    ring = cv2.subtract(outer, mask)
    return ring


def compute_local_features(image_rgb: np.ndarray, region_mask: np.ndarray, product_mask: np.ndarray | None = None) -> dict:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    ring = surrounding_ring(region_mask, pad=12)
    vals = gray[ring > 0]
    if vals.size == 0:
        vals = gray[region_mask == 0]
    gradx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gradx ** 2 + grady ** 2)
    gvals = mag[ring > 0]
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lvals = np.abs(lap)[ring > 0]
    overlap = 0.0
    if product_mask is not None and product_mask.max() > 0:
        overlap = float(((region_mask > 0) & (product_mask > 0)).sum()) / max(1, int((region_mask > 0).sum()))
    return {
        "gray_var": float(np.var(vals)) if vals.size else 0.0,
        "grad_mean": float(np.mean(gvals)) if gvals.size else 0.0,
        "lap_mean": float(np.mean(lvals)) if lvals.size else 0.0,
        "product_overlap": overlap,
    }


def classify_region_complexity(features: dict) -> tuple[str, dict]:
    # Conservative thresholds; hard gate handles residual mistakes later.
    if features["product_overlap"] > 0.15:
        return "complex", {"reason": "region_overlaps_product", **features}
    if features["lap_mean"] < 8.0 and features["gray_var"] < 700 and features["grad_mean"] < 10.0:
        return "simple", {"reason": "low_texture_low_gradient", **features}
    if features["lap_mean"] > 18.0 or features["gray_var"] > 1800 or features["grad_mean"] > 22.0:
        return "complex", {"reason": "high_texture_or_gradient", **features}
    return "medium", {"reason": "mid_complexity", **features}
