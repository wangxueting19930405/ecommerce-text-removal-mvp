from __future__ import annotations

import cv2
import numpy as np
from .types import TextGroup


def draw_overlay(image_rgb: np.ndarray, groups: list[TextGroup], selected_group_ids: set[str], protect_mask: np.ndarray, execution_mask: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    # groups
    for g in groups:
        x1, y1, x2, y2 = g.bbox
        color = (0, 200, 0) if g.group_id in selected_group_ids else (120, 120, 120)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, g.group_id, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    # masks
    if execution_mask.max() > 0:
        red = np.zeros_like(out)
        red[:, :, 0] = 255
        alpha = (execution_mask.astype(np.float32) / 255.0 * 0.28)[..., None]
        out = (out * (1 - alpha) + red * alpha).astype(np.uint8)
    if protect_mask.max() > 0:
        blue = np.zeros_like(out)
        blue[:, :, 2] = 255
        alpha = (protect_mask.astype(np.float32) / 255.0 * 0.22)[..., None]
        out = (out * (1 - alpha) + blue * alpha).astype(np.uint8)
    return out
