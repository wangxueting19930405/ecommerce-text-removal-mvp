from __future__ import annotations

import cv2
import numpy as np

OVERWIDE_AUTO_PRODUCT_RATIO_THRESHOLD = 0.35
OVERWIDE_ERODE_KERNEL_SIZE = 15
OVERWIDE_MAX_ERODE_ITERATIONS = 5


def detect_product_body_mask(image_rgb: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (int(w * 0.08), int(h * 0.10), int(w * 0.84), int(h * 0.82))
    try:
        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8), connectivity=8)
        if num_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            out = np.where(labels == largest, 255, 0).astype(np.uint8)
            return out
        return fg
    except Exception:
        return np.zeros((h, w), np.uint8)


def detect_badge_candidates(image_rgb: np.ndarray, product_mask: np.ndarray | None = None) -> np.ndarray:
    """Lightweight MVP badge protect.

    Purposefully conservative: catch small colorful circular/compact promo badges.
    This is NOT a full icon/logo detector.
    """
    h, w = image_rgb.shape[:2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hch, sch, vch = cv2.split(hsv)
    sat = sch.astype(np.float32)
    val = vch.astype(np.float32)

    # High-saturation, reasonably bright/dark colored decorative elements.
    base = np.where((sat > 85) & (val > 60), 255, 0).astype(np.uint8)
    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    out = np.zeros((h, w), np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((base > 0).astype(np.uint8), connectivity=8)
    img_area = float(h * w)
    upper_limit_y = int(h * 0.78)
    for lab in range(1, num_labels):
        x = stats[lab, cv2.CC_STAT_LEFT]
        y = stats[lab, cv2.CC_STAT_TOP]
        ww = stats[lab, cv2.CC_STAT_WIDTH]
        hh = stats[lab, cv2.CC_STAT_HEIGHT]
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < max(120, int(img_area * 0.00025)):
            continue
        if area > int(img_area * 0.06):
            continue
        if y > upper_limit_y:
            continue
        if x <= 1 or y <= 1 or x + ww >= w - 1 or y + hh >= h - 1:
            continue
        ar = ww / max(1.0, float(hh))
        if not (0.55 <= ar <= 1.8):
            continue
        comp = (labels == lab).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        cnt_area = cv2.contourArea(c)
        circularity = 4.0 * np.pi * cnt_area / max(1.0, peri * peri)
        hull = cv2.convexHull(c)
        hull_area = max(1.0, cv2.contourArea(hull))
        solidity = cnt_area / hull_area
        # compact circular or rounded badge-like objects only
        if circularity < 0.28 and solidity < 0.82:
            continue
        if product_mask is not None and product_mask.max() > 0:
            overlap = float(((comp > 0) & (product_mask > 0)).sum()) / max(1.0, float((comp > 0).sum()))
            if overlap > 0.45:
                continue
        out[comp > 0] = 255

    if out.max() > 0:
        out = cv2.dilate(out, np.ones((5, 5), np.uint8), iterations=1)
    return out


def _mask_pixels(mask: np.ndarray) -> int:
    return int((mask > 0).sum())


def _mask_ratio(mask: np.ndarray, image_pixels: int) -> float:
    return float(_mask_pixels(mask)) / max(1, image_pixels)


def _guard_overwide_auto_product(auto_product: np.ndarray) -> tuple[np.ndarray, dict]:
    image_pixels = int(auto_product.shape[0] * auto_product.shape[1])
    ratio_before = _mask_ratio(auto_product, image_pixels)
    pixels_before = _mask_pixels(auto_product)

    if ratio_before <= OVERWIDE_AUTO_PRODUCT_RATIO_THRESHOLD:
        return auto_product, {
            "auto_product_pixels_before": pixels_before,
            "auto_product_pixels_after": pixels_before,
            "auto_product_ratio_before": ratio_before,
            "auto_product_ratio_after": ratio_before,
            "overwide_triggered": False,
            "overwide_action": "none",
            "erode_iterations": 0,
        }

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (OVERWIDE_ERODE_KERNEL_SIZE, OVERWIDE_ERODE_KERNEL_SIZE),
    )
    adjusted = auto_product.copy()
    erode_iterations = 0
    for _ in range(OVERWIDE_MAX_ERODE_ITERATIONS):
        adjusted = cv2.erode(adjusted, kernel, iterations=1)
        erode_iterations += 1
        ratio_after = _mask_ratio(adjusted, image_pixels)
        if ratio_after <= OVERWIDE_AUTO_PRODUCT_RATIO_THRESHOLD:
            return adjusted, {
                "auto_product_pixels_before": pixels_before,
                "auto_product_pixels_after": _mask_pixels(adjusted),
                "auto_product_ratio_before": ratio_before,
                "auto_product_ratio_after": ratio_after,
                "overwide_triggered": True,
                "overwide_action": "eroded",
                "erode_iterations": erode_iterations,
            }

    disabled = np.zeros_like(auto_product)
    return disabled, {
        "auto_product_pixels_before": pixels_before,
        "auto_product_pixels_after": 0,
        "auto_product_ratio_before": ratio_before,
        "auto_product_ratio_after": 0.0,
        "overwide_triggered": True,
        "overwide_action": "disabled",
        "erode_iterations": erode_iterations,
    }


def build_protect_mask(image_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    auto_product_raw = detect_product_body_mask(image_rgb)
    auto_product, overwide_debug = _guard_overwide_auto_product(auto_product_raw)
    auto_badge = detect_badge_candidates(image_rgb, product_mask=auto_product)
    protect = cv2.bitwise_or(auto_product, auto_badge)
    return protect, {
        "phase": "phase1",
        "phase_scope": "product_body_auto_and_lightweight_badge_candidates_only",
        "auto_product_pixels": _mask_pixels(auto_product),
        "auto_badge_pixels": _mask_pixels(auto_badge),
        "protect_pixels": _mask_pixels(protect),
        "auto_product_ratio_before": overwide_debug["auto_product_ratio_before"],
        "auto_product_ratio_after": overwide_debug["auto_product_ratio_after"],
        "overwide_triggered": overwide_debug["overwide_triggered"],
        "overwide_action": overwide_debug["overwide_action"],
        "erode_iterations": overwide_debug["erode_iterations"],
        "badge_candidate_enabled": True,
        "manual_user_mask_supported": False,
        "protect_sources": {
            "product_body_auto": {
                "enabled": True,
                "pixels": _mask_pixels(auto_product),
                "pixels_before": overwide_debug["auto_product_pixels_before"],
                "pixels_after": overwide_debug["auto_product_pixels_after"],
                "ratio_before": overwide_debug["auto_product_ratio_before"],
                "ratio_after": overwide_debug["auto_product_ratio_after"],
                "overwide_triggered": overwide_debug["overwide_triggered"],
                "overwide_action": overwide_debug["overwide_action"],
                "erode_iterations": overwide_debug["erode_iterations"],
            },
            "badge_candidate_lightweight": {
                "enabled": True,
                "pixels": _mask_pixels(auto_badge),
            },
        },
    }
