from __future__ import annotations

import cv2
import numpy as np
from .complexity import surrounding_ring


def local_hard_gate(orig_rgb: np.ndarray, cand_rgb: np.ndarray, region_mask: np.ndarray, protect_mask: np.ndarray) -> tuple[bool, list[str], dict]:
    reasons: list[str] = []
    metrics: dict = {}
    # 1) protect damage
    if protect_mask.max() > 0:
        diff = np.abs(cand_rgb.astype(np.int16) - orig_rgb.astype(np.int16)).mean(axis=2)
        pd = float(diff[protect_mask > 0].mean()) if (protect_mask > 0).any() else 0.0
        metrics['protect_damage_mean'] = pd
        if pd > 8.0:
            reasons.append('protect_damage')
    # 2) edge/ring discontinuity
    ring = surrounding_ring(region_mask, pad=4)
    gray_o = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    gray_c = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2GRAY)
    ring_diff = float(np.abs(gray_c.astype(np.int16) - gray_o.astype(np.int16))[ring > 0].mean()) if (ring > 0).any() else 0.0
    metrics['ring_diff_mean'] = ring_diff
    if ring_diff > 22.0:
        reasons.append('ring_discontinuity')
    # 3) simple residual proxy: strong edges still inside region after fill
    gradx = cv2.Sobel(gray_c, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray_c, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gradx ** 2 + grady ** 2)
    residual = float(np.mean(mag[region_mask > 0])) if (region_mask > 0).any() else 0.0
    metrics['residual_grad_mean'] = residual
    if residual > 26.0:
        reasons.append('residual_or_texture_streak')

    return len(reasons) == 0, reasons, metrics
