from __future__ import annotations

import cv2
import numpy as np
from .ocr_google import detect_google_vision_boxes
from .types import OCRBox, OCRResult, TextGroup


def detect_heuristic_boxes(image_rgb: np.ndarray) -> tuple[list[OCRBox], dict]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # detect both dark-on-light and light-on-dark text-like structures
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    merged = cv2.max(blackhat, tophat)
    _, thr = cv2.threshold(merged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((thr > 0).astype(np.uint8), connectivity=8)
    boxes: list[OCRBox] = []
    for lab in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[lab]]
        if area < 40 or w < 6 or h < 6:
            continue
        if w / max(h, 1) < 0.3 or w / max(h, 1) > 30:
            continue
        poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        boxes.append(OCRBox(polygon=poly, bbox=(x, y, x + w, y + h), text="", confidence=0.0))
    return boxes, {
        "ok": True,
        "available": True,
        "source": "heuristic",
        "box_count": len(boxes),
    }


def group_boxes(boxes: list[OCRBox]) -> list[TextGroup]:
    # lightweight spatial grouping; keeps user-facing units understandable without heavy semantics
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i].bbox[1], boxes[i].bbox[0]))
    groups: list[TextGroup] = []
    current: list[int] = []
    def flush():
        nonlocal current
        if not current:
            return
        xs1 = [boxes[i].bbox[0] for i in current]
        ys1 = [boxes[i].bbox[1] for i in current]
        xs2 = [boxes[i].bbox[2] for i in current]
        ys2 = [boxes[i].bbox[3] for i in current]
        gid = f"g{len(groups)+1:03d}"
        text = " ".join((boxes[i].text or "").strip() for i in current if (boxes[i].text or "").strip())
        groups.append(TextGroup(gid, current[:], (min(xs1), min(ys1), max(xs2), max(ys2)), text))
        current = []

    last = None
    for idx in order:
        b = boxes[idx].bbox
        if last is None:
            current.append(idx)
            last = b
            continue
        lx1, ly1, lx2, ly2 = last
        x1, y1, x2, y2 = b
        last_h = max(1, ly2 - ly1)
        # same line-ish and close horizontally
        if abs(y1 - ly1) <= max(12, int(0.6 * last_h)) and x1 - lx2 <= max(30, int(0.8 * last_h)):
            current.append(idx)
            last = (min(lx1, x1), min(ly1, y1), max(lx2, x2), max(ly2, y2))
        else:
            flush()
            current.append(idx)
            last = b
    flush()
    return groups


def _build_ocr_result(
    boxes: list[OCRBox],
    provider: str,
    degraded: bool,
    text_reliable: bool,
    debug: dict,
) -> OCRResult:
    final_debug = dict(debug)
    final_debug["ocr_provider"] = provider
    final_debug["ocr_degraded"] = degraded
    final_debug["text_reliable"] = text_reliable
    return OCRResult(
        boxes=boxes,
        groups=group_boxes(boxes),
        provider=provider,
        degraded=degraded,
        text_reliable=text_reliable,
        debug=final_debug,
    )


def run_ocr(image_rgb: np.ndarray, prefer_google_ocr: bool = True) -> OCRResult:
    if prefer_google_ocr:
        google_boxes, google_debug = detect_google_vision_boxes(image_rgb)
        if google_debug.get("ok", False):
            google_debug = dict(google_debug)
            google_debug["fallback_used"] = False
            return _build_ocr_result(
                boxes=google_boxes,
                provider="google_vision",
                degraded=False,
                text_reliable=True,
                debug=google_debug,
            )

        heuristic_boxes, heuristic_debug = detect_heuristic_boxes(image_rgb)
        heuristic_debug = dict(heuristic_debug)
        heuristic_debug["fallback_used"] = True
        heuristic_debug["google_vision_attempt"] = google_debug
        return _build_ocr_result(
            boxes=heuristic_boxes,
            provider="heuristic",
            degraded=True,
            text_reliable=False,
            debug=heuristic_debug,
        )

    heuristic_boxes, heuristic_debug = detect_heuristic_boxes(image_rgb)
    heuristic_debug = dict(heuristic_debug)
    heuristic_debug["fallback_used"] = False
    heuristic_debug["google_vision_attempt"] = {
        "ok": False,
        "available": False,
        "reason": "google_vision_disabled",
    }
    return _build_ocr_result(
        boxes=heuristic_boxes,
        provider="heuristic",
        degraded=True,
        text_reliable=False,
        debug=heuristic_debug,
    )
