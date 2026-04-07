from __future__ import annotations

import binascii
import io
import shutil
import unittest
import uuid
from pathlib import Path
from urllib.parse import urlparse
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from pipeline_v2.lama_adapter import run_lama_like
from tools.lama_server.app import app, get_backend


def _png_bytes_from_rgb(rgb: np.ndarray) -> bytes:
    bio = io.BytesIO()
    Image.fromarray(rgb.astype("uint8"), "RGB").save(bio, format="PNG")
    return bio.getvalue()


def _png_bytes_from_mask(mask: np.ndarray) -> bytes:
    bio = io.BytesIO()
    Image.fromarray(mask.astype("uint8"), "L").save(bio, format="PNG")
    return bio.getvalue()


class _FakeBackend:
    backend_name = "fake_lama_backend"
    uses_protect_mask = False

    def restore(
        self,
        image_rgb: np.ndarray,
        execution_mask: np.ndarray,
        protect_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        del protect_mask
        result = image_rgb.copy()
        result[execution_mask > 0] = np.array([0, 255, 0], dtype=np.uint8)
        return result


class LaMaServerTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[get_backend] = lambda: _FakeBackend()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])

    def test_restore_matches_adapter_protocol_and_returns_best_image_hex(self) -> None:
        image = np.full((8, 8, 3), 255, np.uint8)
        execution_mask = np.zeros((8, 8), np.uint8)
        execution_mask[2:6, 2:6] = 255
        protect_mask = np.zeros((8, 8), np.uint8)

        response = self.client.post(
            "/restore",
            files={
                "image": ("image.png", _png_bytes_from_rgb(image), "image/png"),
                "execution_mask": ("execution_mask.png", _png_bytes_from_mask(execution_mask), "image/png"),
                "protect_mask": ("protect_mask.png", _png_bytes_from_mask(protect_mask), "image/png"),
            },
            data={
                "scene_type": "panel",
                "route": "lama_route",
                "num_candidates": "1",
                "return_all_candidates": "0",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("best_image_hex", payload)
        restored = np.array(Image.open(io.BytesIO(binascii.unhexlify(payload["best_image_hex"]))).convert("RGB"))
        self.assertEqual(restored.shape, image.shape)
        self.assertEqual(payload["meta"]["backend"], "fake_lama_backend")
        self.assertTrue(payload["meta"]["used_execution_mask"])
        self.assertFalse(payload["meta"]["used_protect_mask"])
        self.assertEqual(payload["meta"]["scene_type"], "panel")
        self.assertEqual(payload["meta"]["route"], "lama_route")

    def test_restore_requires_image(self) -> None:
        execution_mask = np.zeros((8, 8), np.uint8)
        response = self.client.post(
            "/restore",
            files={
                "execution_mask": ("execution_mask.png", _png_bytes_from_mask(execution_mask), "image/png"),
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "missing_required_file:image")

    def test_restore_requires_execution_mask(self) -> None:
        image = np.full((8, 8, 3), 255, np.uint8)
        response = self.client.post(
            "/restore",
            files={
                "image": ("image.png", _png_bytes_from_rgb(image), "image/png"),
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "missing_required_file:execution_mask")

    def test_restore_rejects_size_mismatch(self) -> None:
        image = np.full((8, 8, 3), 255, np.uint8)
        execution_mask = np.zeros((6, 6), np.uint8)
        response = self.client.post(
            "/restore",
            files={
                "image": ("image.png", _png_bytes_from_rgb(image), "image/png"),
                "execution_mask": ("execution_mask.png", _png_bytes_from_mask(execution_mask), "image/png"),
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "image_execution_mask_size_mismatch")

    def test_current_lama_adapter_can_call_thin_service_without_protocol_changes(self) -> None:
        image = np.full((8, 8, 3), 255, np.uint8)
        execution_mask = np.zeros((8, 8), np.uint8)
        execution_mask[2:6, 2:6] = 255
        protect_mask = np.zeros((8, 8), np.uint8)

        class _ResponseShim:
            def __init__(self, response):
                self._response = response

            def raise_for_status(self) -> None:
                self._response.raise_for_status()

            def json(self):
                return self._response.json()

        def _fake_post(url, files=None, data=None, timeout=None):
            del timeout
            multipart_files = {}
            for key, value in (files or {}).items():
                filename, handle, content_type = value
                multipart_files[key] = (filename, handle.read(), content_type)
            response = self.client.post(urlparse(url).path, files=multipart_files, data=data)
            return _ResponseShim(response)

        class _WorkspaceTempDir:
            def __init__(self) -> None:
                self.path = Path("tests") / f"_tmp_lama_adapter_{uuid.uuid4().hex}"

            def __enter__(self) -> str:
                self.path.mkdir(parents=True, exist_ok=True)
                return str(self.path)

            def __exit__(self, exc_type, exc, tb) -> None:
                shutil.rmtree(self.path, ignore_errors=True)

        with patch("pipeline_v2.lama_adapter.requests.post", side_effect=_fake_post), \
             patch("pipeline_v2.lama_adapter.tempfile.TemporaryDirectory", _WorkspaceTempDir):
            restored_rgb, restore_debug = run_lama_like(
                image_rgb=image,
                region_mask=execution_mask,
                protect_mask=protect_mask,
                external_url="http://127.0.0.1:8009",
                require_lama=True,
                allow_fallback=False,
            )

        self.assertEqual(restore_debug["method"], "external_lama")
        self.assertTrue(restore_debug["backend_ok"])
        self.assertEqual(restored_rgb.shape, image.shape)


class _FakeBackendBadShape:
    """Returns an ndarray whose shape does NOT match input — simulates a backend
    that forgot to crop padding AND also skips the inference-layer normalization
    (used to verify the app-layer guard still triggers when bypass happens)."""

    backend_name = "fake_bad_shape"
    uses_protect_mask = False

    def __init__(self, out_h: int, out_w: int) -> None:
        self._out_h = out_h
        self._out_w = out_w

    def restore(
        self,
        image_rgb: np.ndarray,
        execution_mask: np.ndarray,
        protect_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        del image_rgb, execution_mask, protect_mask
        return np.zeros((self._out_h, self._out_w, 3), dtype=np.uint8)


class LaMaInferenceCropTests(unittest.TestCase):
    """Unit-tests for the padding-crop logic added to SimpleLaMaBackend.restore()."""

    def _make_backend_with_fake_model(self, model_callable):
        """Return a SimpleLaMaBackend whose _get_model() yields model_callable."""
        from tools.lama_server.inference import SimpleLaMaBackend
        backend = SimpleLaMaBackend.__new__(SimpleLaMaBackend)
        backend._model = model_callable
        return backend

    def test_exact_size_passes_through(self) -> None:
        """Backend output == input size → no crop, returned as-is."""
        inp = np.full((10, 15, 3), 128, dtype=np.uint8)
        mask = np.zeros((10, 15), dtype=np.uint8)

        def fake_model(*_):
            return Image.fromarray(np.full((10, 15, 3), 200, dtype=np.uint8), "RGB")

        backend = self._make_backend_with_fake_model(fake_model)
        result = backend.restore(inp, mask)
        self.assertEqual(result.shape, (10, 15, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_padded_output_is_cropped_back(self) -> None:
        """Backend output larger than input (padding) → crop to original size."""
        inp = np.full((10, 15, 3), 128, dtype=np.uint8)
        mask = np.zeros((10, 15), dtype=np.uint8)

        # Simulate LaMa padding: output is 16x16 even though input was 10x15
        def fake_model(*_):
            return Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8), "RGB")

        backend = self._make_backend_with_fake_model(fake_model)
        result = backend.restore(inp, mask)
        self.assertEqual(result.shape, (10, 15, 3), "should have been cropped back to input size")
        self.assertEqual(result.dtype, np.uint8)

    def test_output_smaller_than_input_raises(self) -> None:
        """Backend output smaller than input → LaMaBackendError, no resize."""
        from tools.lama_server.inference import LaMaBackendError
        inp = np.full((20, 20, 3), 128, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)

        def fake_model(*_):
            return Image.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8), "RGB")

        backend = self._make_backend_with_fake_model(fake_model)
        with self.assertRaises(LaMaBackendError) as ctx:
            backend.restore(inp, mask)
        self.assertIn("smaller_than_input", str(ctx.exception))


class LaMaAppBadOutputTests(unittest.TestCase):
    """Verify app-layer 500 guard and its log fields when inference somehow still
    returns wrong shape (e.g. future backend bypass)."""

    def setUp(self) -> None:
        app.dependency_overrides[get_backend] = lambda: _FakeBackendBadShape(16, 16)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_bad_output_shape_returns_500(self) -> None:
        """If backend returns wrong shape the app must return 500, not 200."""
        image = np.full((10, 10, 3), 255, np.uint8)
        execution_mask = np.zeros((10, 10), np.uint8)

        response = self.client.post(
            "/restore",
            files={
                "image": ("image.png", _png_bytes_from_rgb(image), "image/png"),
                "execution_mask": ("execution_mask.png", _png_bytes_from_mask(execution_mask), "image/png"),
            },
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"], "invalid_backend_output")


if __name__ == "__main__":
    unittest.main()
