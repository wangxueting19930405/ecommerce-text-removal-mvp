from __future__ import annotations

import os

import cv2
import numpy as np

from .types import OCRBox


def _bbox_from_polygon(poly: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def detect_google_vision_boxes(image_rgb: np.ndarray) -> tuple[list[OCRBox], dict]:
    try:
        from google.cloud import vision  # type: ignore
    except Exception as exc:
        return [], {
            "ok": False,
            "available": False,
            "reason": f"google_import_failed:{exc}",
        }

    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        return [], {
            "ok": False,
            "available": False,
            "reason": "missing_google_credentials",
        }

    try:
        client = vision.ImageAnnotatorClient()
        ok, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            return [], {
                "ok": False,
                "available": False,
                "reason": "encode_failed",
            }

        image = vision.Image(content=buf.tobytes())
        response = client.text_detection(image=image)

        boxes: list[OCRBox] = []
        anns = response.text_annotations or []
        for ann in anns[1:]:
            verts = ann.bounding_poly.vertices
            poly = [(int(v.x or 0), int(v.y or 0)) for v in verts]
            if len(poly) < 4:
                continue
            bbox = _bbox_from_polygon(poly)
            boxes.append(OCRBox(
                polygon=poly,
                bbox=bbox,
                text=ann.description or "",
                confidence=0.0,
            ))

        return boxes, {
            "ok": True,
            "available": True,
            "source": "google_vision",
            "credentials_env": "GOOGLE_APPLICATION_CREDENTIALS",
            "box_count": len(boxes),
        }
    except Exception as exc:
        return [], {
            "ok": False,
            "available": False,
            "reason": f"google_runtime_failed:{exc}",
        }
