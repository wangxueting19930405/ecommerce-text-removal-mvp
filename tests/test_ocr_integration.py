from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline_v2.ocr import run_ocr
from pipeline_v2.service import analyze_image, remove_text_image
from pipeline_v2.types import OCRBox, OCRResult, TextGroup


def _blank_protect(image_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    shape = image_rgb.shape[:2]
    return np.zeros(shape, np.uint8), {
        "phase": "phase1",
        "phase_scope": "product_body_auto_and_lightweight_badge_candidates_only",
        "auto_product_pixels": 0,
        "auto_badge_pixels": 0,
        "protect_pixels": 0,
        "auto_product_ratio_before": 0.0,
        "auto_product_ratio_after": 0.0,
        "overwide_triggered": False,
        "overwide_action": "none",
        "erode_iterations": 0,
        "badge_candidate_enabled": True,
        "manual_user_mask_supported": False,
        "protect_sources": {
            "product_body_auto": {
                "enabled": True,
                "pixels": 0,
                "pixels_before": 0,
                "pixels_after": 0,
                "ratio_before": 0.0,
                "ratio_after": 0.0,
                "overwide_triggered": False,
                "overwide_action": "none",
                "erode_iterations": 0,
            },
            "badge_candidate_lightweight": {"enabled": True, "pixels": 0},
        },
    }


def _google_boxes() -> tuple[list[OCRBox], dict]:
    boxes = [
        OCRBox(
            polygon=[(10, 10), (40, 10), (40, 28), (10, 28)],
            bbox=(10, 10, 40, 28),
            text="summer",
        ),
        OCRBox(
            polygon=[(45, 10), (78, 10), (78, 28), (45, 28)],
            bbox=(45, 10, 78, 28),
            text="sale",
        ),
    ]
    return boxes, {
        "ok": True,
        "available": True,
        "source": "google_vision",
        "box_count": len(boxes),
    }


def _heuristic_boxes() -> tuple[list[OCRBox], dict]:
    boxes = [
        OCRBox(
            polygon=[(10, 10), (40, 10), (40, 28), (10, 28)],
            bbox=(10, 10, 40, 28),
            text="",
        ),
        OCRBox(
            polygon=[(45, 10), (78, 10), (78, 28), (45, 28)],
            bbox=(45, 10, 78, 28),
            text="",
        ),
    ]
    return boxes, {
        "ok": True,
        "available": True,
        "source": "heuristic",
        "box_count": len(boxes),
    }


class OCRIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((96, 128, 3), 255, np.uint8)

    @patch("pipeline_v2.ocr.detect_google_vision_boxes")
    def test_google_success_path_marks_provider_and_text_reliability(self, mock_google) -> None:
        boxes, debug = _google_boxes()
        mock_google.return_value = (boxes, debug)

        result = run_ocr(self.image, prefer_google_ocr=True)

        self.assertEqual(result.provider, "google_vision")
        self.assertFalse(result.degraded)
        self.assertTrue(result.text_reliable)
        self.assertGreaterEqual(len(result.groups), 1)
        self.assertTrue(all(group.text for group in result.groups))

    @patch("pipeline_v2.ocr.detect_heuristic_boxes")
    @patch("pipeline_v2.ocr.detect_google_vision_boxes")
    def test_google_failure_falls_back_to_heuristic(self, mock_google, mock_heuristic) -> None:
        mock_google.return_value = ([], {
            "ok": False,
            "available": False,
            "reason": "missing_google_credentials",
        })
        h_boxes, h_debug = _heuristic_boxes()
        mock_heuristic.return_value = (h_boxes, h_debug)

        result = run_ocr(self.image, prefer_google_ocr=True)

        self.assertEqual(result.provider, "heuristic")
        self.assertTrue(result.degraded)
        self.assertFalse(result.text_reliable)
        self.assertEqual(result.debug["google_vision_attempt"]["reason"], "missing_google_credentials")

    def test_analyze_and_remove_text_share_same_ocr_semantics(self) -> None:
        boxes, debug = _google_boxes()
        ocr_result = OCRResult(
            boxes=boxes,
            groups=[
                TextGroup(
                    group_id="g001",
                    box_indices=[0, 1],
                    bbox=(10, 10, 78, 28),
                    text="summer sale",
                )
            ],
            provider="google_vision",
            degraded=False,
            text_reliable=True,
            debug={
                **debug,
                "ocr_provider": "google_vision",
                "ocr_degraded": False,
                "text_reliable": True,
            },
        )

        with patch("pipeline_v2.service.run_ocr", return_value=ocr_result), \
             patch("pipeline_v2.service.build_protect_mask", side_effect=_blank_protect):
            analyze_result = analyze_image(self.image)
            selected_path = Path("tests") / "_tmp_selected_groups.json"
            try:
                selected_path.write_text(json.dumps({"selected_group_ids": []}), encoding="utf-8")
                remove_result = remove_text_image(
                    self.image,
                    selected_groups_json_path=selected_path,
                    require_lama=False,
                    allow_lama_fallback=True,
                )
            finally:
                if selected_path.exists():
                    selected_path.unlink()

        self.assertEqual(analyze_result["ocr_provider"], "google_vision")
        self.assertFalse(analyze_result["ocr_degraded"])
        self.assertTrue(all(group["text_reliable"] for group in analyze_result["groups"]))
        self.assertEqual(remove_result.diagnostics["ocr_provider"], "google_vision")
        self.assertFalse(remove_result.diagnostics["ocr_degraded"])
        self.assertTrue(all(group["text_reliable"] for group in remove_result.diagnostics["groups"]))

    @patch("pipeline_v2.service.build_protect_mask", side_effect=_blank_protect)
    def test_remove_text_mainline_still_runs_with_degraded_ocr(self, _mock_protect) -> None:
        result = remove_text_image(
            self.image,
            prefer_google_ocr=False,
            require_lama=False,
            allow_lama_fallback=True,
        )

        self.assertEqual(result.output_rgb.shape, self.image.shape)
        self.assertEqual(result.overlay_rgb.shape, self.image.shape)
        self.assertEqual(result.diagnostics["ocr_provider"], "heuristic")
        self.assertTrue(result.diagnostics["ocr_degraded"])
        self.assertIn("region_decisions", result.diagnostics)


if __name__ == "__main__":
    unittest.main()
