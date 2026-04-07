from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from pipeline_v2.protect import (
    OVERWIDE_AUTO_PRODUCT_RATIO_THRESHOLD,
    OVERWIDE_MAX_ERODE_ITERATIONS,
    build_protect_mask,
)


def _rect_mask(shape: tuple[int, int], x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


class ProtectOverwideTests(unittest.TestCase):
    def test_none_when_auto_product_ratio_is_within_threshold(self) -> None:
        image = np.zeros((100, 100, 3), np.uint8)
        auto_product = _rect_mask((100, 100), 10, 10, 60, 60)
        auto_badge = _rect_mask((100, 100), 80, 5, 90, 15)

        with patch("pipeline_v2.protect.detect_product_body_mask", return_value=auto_product), \
             patch("pipeline_v2.protect.detect_badge_candidates", return_value=auto_badge):
            protect_mask, debug = build_protect_mask(image)

        self.assertFalse(debug["overwide_triggered"])
        self.assertEqual(debug["overwide_action"], "none")
        self.assertEqual(debug["erode_iterations"], 0)
        self.assertLessEqual(debug["auto_product_ratio_before"], OVERWIDE_AUTO_PRODUCT_RATIO_THRESHOLD)
        self.assertEqual(debug["auto_product_ratio_before"], debug["auto_product_ratio_after"])
        self.assertEqual(debug["auto_product_pixels"], int((auto_product > 0).sum()))
        self.assertEqual(debug["auto_badge_pixels"], int((auto_badge > 0).sum()))
        self.assertGreater(int((protect_mask > 0).sum()), debug["auto_product_pixels"])

    def test_eroded_when_auto_product_can_be_brought_back_under_threshold(self) -> None:
        image = np.zeros((100, 100, 3), np.uint8)
        auto_product = _rect_mask((100, 100), 10, 10, 90, 60)

        with patch("pipeline_v2.protect.detect_product_body_mask", return_value=auto_product), \
             patch("pipeline_v2.protect.detect_badge_candidates", return_value=np.zeros((100, 100), np.uint8)):
            protect_mask, debug = build_protect_mask(image)

        self.assertTrue(debug["overwide_triggered"])
        self.assertEqual(debug["overwide_action"], "eroded")
        self.assertGreater(debug["erode_iterations"], 0)
        self.assertGreater(debug["auto_product_ratio_before"], OVERWIDE_AUTO_PRODUCT_RATIO_THRESHOLD)
        self.assertLessEqual(debug["auto_product_ratio_after"], OVERWIDE_AUTO_PRODUCT_RATIO_THRESHOLD)
        self.assertEqual(int((protect_mask > 0).sum()), debug["auto_product_pixels"])
        self.assertGreater(debug["auto_product_pixels"], 0)

    def test_disabled_when_auto_product_stays_overwide_after_max_erosion(self) -> None:
        image = np.zeros((400, 400, 3), np.uint8)
        auto_product = np.full((400, 400), 255, np.uint8)
        auto_badge = _rect_mask((400, 400), 10, 10, 40, 40)

        with patch("pipeline_v2.protect.detect_product_body_mask", return_value=auto_product), \
             patch("pipeline_v2.protect.detect_badge_candidates", return_value=auto_badge):
            protect_mask, debug = build_protect_mask(image)

        self.assertTrue(debug["overwide_triggered"])
        self.assertEqual(debug["overwide_action"], "disabled")
        self.assertEqual(debug["erode_iterations"], OVERWIDE_MAX_ERODE_ITERATIONS)
        self.assertGreater(debug["auto_product_ratio_before"], OVERWIDE_AUTO_PRODUCT_RATIO_THRESHOLD)
        self.assertEqual(debug["auto_product_ratio_after"], 0.0)
        self.assertEqual(debug["auto_product_pixels"], 0)
        self.assertEqual(int((protect_mask > 0).sum()), int((auto_badge > 0).sum()))
        self.assertEqual(debug["auto_badge_pixels"], int((auto_badge > 0).sum()))
        self.assertEqual(debug["protect_sources"]["product_body_auto"]["overwide_action"], "disabled")


if __name__ == "__main__":
    unittest.main()
