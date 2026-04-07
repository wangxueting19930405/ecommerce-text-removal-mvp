from __future__ import annotations

import cv2
import numpy as np
from .types import OCRBox, TextGroup


def rasterize_selected_groups(shape: tuple[int, int], boxes: list[OCRBox], groups: list[TextGroup], selected_group_ids: set[str]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    selected = [g for g in groups if g.group_id in selected_group_ids]
    for g in selected:
        for bi in g.box_indices:
            pts = np.array(boxes[bi].polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


def estimate_dilation(mask: np.ndarray) -> int:
    # Simple and intentionally conservative for MVP.
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    hs = []
    for lab in range(1, num_labels):
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        if h > 0:
            hs.append(h)
    median_h = np.median(hs) if hs else 12
    return int(np.clip(round(median_h * 0.18), 5, 10))


def build_execution_mask(shape: tuple[int, int], boxes: list[OCRBox], groups: list[TextGroup], selected_group_ids: set[str], protect_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    base = rasterize_selected_groups(shape, boxes, groups, selected_group_ids)
    dilation = estimate_dilation(base)
    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
        dilated = cv2.dilate(base, kernel, iterations=1)
    else:
        dilated = base.copy()
    execution = cv2.bitwise_and(dilated, cv2.bitwise_not(protect_mask))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((execution > 0).astype(np.uint8), connectivity=8)
    regions = np.zeros_like(execution)
    for lab in range(1, num_labels):
        if stats[lab, cv2.CC_STAT_AREA] >= 20:
            regions[labels == lab] = 255
    return base, regions, {
        "selection_pixels": int((base > 0).sum()),
        "dilation_px": dilation,
        "execution_pixels": int((regions > 0).sum()),
        "region_count": int(max(0, num_labels - 1)),
    }


def iter_regions(execution_mask: np.ndarray):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((execution_mask > 0).astype(np.uint8), connectivity=8)
    for lab in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[lab]]
        if area < 20:
            continue
        region_mask = np.where(labels == lab, 255, 0).astype(np.uint8)
        yield lab, (x, y, x + w, y + h), region_mask
