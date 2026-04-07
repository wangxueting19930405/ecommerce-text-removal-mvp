from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

import numpy as np
from PIL import Image

LOGGER = logging.getLogger("lama_server")


class LaMaBackendError(RuntimeError):
    pass


class LaMaInferenceBackend(Protocol):
    backend_name: str
    uses_protect_mask: bool

    def restore(
        self,
        image_rgb: np.ndarray,
        execution_mask: np.ndarray,
        protect_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        ...


class SimpleLaMaBackend:
    backend_name = "simple_lama_inpainting"
    uses_protect_mask = False

    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model

        SimpleLama = None
        import_errors: list[str] = []
        for module_name in ("simple_lama_inpainting", "simple_lama"):
            try:
                module = __import__(module_name, fromlist=["SimpleLama"])
                SimpleLama = getattr(module, "SimpleLama")
                self.backend_name = module_name
                break
            except Exception as exc:
                import_errors.append(f"{module_name}:{exc}")

        if SimpleLama is None:
            raise LaMaBackendError(
                "simple_lama_backend_not_installed; install requirements-lama.txt first; "
                + " | ".join(import_errors)
            )

        try:
            self._model = SimpleLama()
        except Exception as exc:
            raise LaMaBackendError(f"simple_lama_initialization_failed:{exc}") from exc

        return self._model

    def restore(
        self,
        image_rgb: np.ndarray,
        execution_mask: np.ndarray,
        protect_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        del protect_mask

        model = self._get_model()
        image = Image.fromarray(image_rgb.astype("uint8"), "RGB")
        mask = Image.fromarray(execution_mask.astype("uint8"), "L")

        try:
            result = model(image, mask)
        except Exception as exc:
            raise LaMaBackendError(f"simple_lama_inference_failed:{exc}") from exc

        if not isinstance(result, Image.Image):
            raise LaMaBackendError("simple_lama_invalid_result_type")

        result_arr = np.array(result.convert("RGB"))

        # Normalize dtype: LaMa backends occasionally return float32 [0,255]
        if result_arr.dtype != np.uint8:
            result_arr = result_arr.astype(np.uint8)

        # LaMa pads input to stride multiples before inference and may not
        # crop back. If output is larger than input, crop to original size.
        orig_h, orig_w = image_rgb.shape[:2]
        out_h, out_w = result_arr.shape[:2]
        if (out_h, out_w) != (orig_h, orig_w):
            if out_h < orig_h or out_w < orig_w:
                raise LaMaBackendError(
                    f"simple_lama_output_smaller_than_input:"
                    f"input=({orig_h},{orig_w}) output=({out_h},{out_w})"
                )
            LOGGER.warning(
                "LaMa output padded: input=(%d,%d) output=(%d,%d); cropping back",
                orig_h, orig_w, out_h, out_w,
            )
            result_arr = result_arr[:orig_h, :orig_w, :]

        return result_arr


@lru_cache(maxsize=1)
def get_default_backend() -> SimpleLaMaBackend:
    return SimpleLaMaBackend()
