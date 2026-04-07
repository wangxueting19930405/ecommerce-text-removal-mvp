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

    # Decide between solid-like fill and inpaint fallback using local color variance.
    ring_pixels = image_rgb[ring > 0]
    color_var = float(np.var(ring_pixels.astype(np.float32), axis=0).mean()) if ring_pixels.size else 0.0
    if color_var < 400:
        fill_color = np.median(ring_pixels, axis=0).astype(np.uint8)
        out[region_mask > 0] = fill_color
        feather = cv2.GaussianBlur(out, (0, 0), 1.2)
        alpha = cv2.GaussianBlur(region_mask, (0, 0), 1.1).astype(np.float32) / 255.0
        alpha = alpha[..., None]
        out = (image_rgb * (1 - alpha) + feather * alpha).astype(np.uint8)
        return out, {"method": "solid_fill"}

    inpaint = cv2.inpaint(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), region_mask, 3, cv2.INPAINT_TELEA)
    out = cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)
    return out, {"method": "opencv_telea"}
