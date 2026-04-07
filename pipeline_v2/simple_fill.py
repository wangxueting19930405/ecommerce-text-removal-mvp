from __future__ import annotations

import cv2
import numpy as np
from .complexity import surrounding_ring


def simple_fill_region(image_rgb: np.ndarray, region_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    ring = surrounding_ring(region_mask, pad=16)
    out = image_rgb.copy()
    ys, xs = np.where(ring > 0)
    if len(xs) == 0:
        inpaint = cv2.inpaint(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), region_mask, 3, cv2.INPAINT_TELEA)
        return cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB), {"method": "opencv_telea_fallback"}

    # Decide between gradient fill and inpaint fallback using local color variance.
    ring_pixels = image_rgb[ring > 0]
    color_var = float(np.var(ring_pixels.astype(np.float32), axis=0).mean()) if ring_pixels.size else 0.0
    if color_var < 400:
        ring_ys, ring_xs = np.where(ring > 0)
        # Fit a linear gradient per channel: color(y, x) = a*x + b*y + c
        # Falls back to solid fill naturally when background is uniform (a≈0, b≈0).
        A = np.column_stack([ring_xs, ring_ys, np.ones(len(ring_xs))]).astype(np.float32)
        reg_ys, reg_xs = np.where(region_mask > 0)
        A_reg = np.column_stack([reg_xs, reg_ys, np.ones(len(reg_xs))]).astype(np.float32)
        for ch in range(3):
            coeffs, _, _, _ = np.linalg.lstsq(A, ring_pixels[:, ch].astype(np.float32), rcond=None)
            predicted = A_reg @ coeffs
            out[reg_ys, reg_xs, ch] = np.clip(predicted, 0, 255).astype(np.uint8)
        feather = cv2.GaussianBlur(out, (0, 0), 1.2)
        alpha = cv2.GaussianBlur(region_mask, (0, 0), 1.1).astype(np.float32) / 255.0
        alpha = alpha[..., None]
        out = (image_rgb * (1 - alpha) + feather * alpha).astype(np.uint8)
        return out, {"method": "gradient_fill"}

    inpaint = cv2.inpaint(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), region_mask, 3, cv2.INPAINT_TELEA)
    out = cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)
    return out, {"method": "opencv_telea"}
