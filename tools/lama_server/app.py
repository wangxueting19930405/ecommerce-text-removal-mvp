from __future__ import annotations

from env_bootstrap import bootstrap_env

bootstrap_env()

import logging

import numpy as np
from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from .inference import LaMaBackendError, LaMaInferenceBackend, get_default_backend
from .utils import encode_png_hex, read_binary_mask_bytes, read_rgb_image_bytes

LOGGER = logging.getLogger("lama_server")

app = FastAPI(title="ecommerce_text_removal_lama_server")


def get_backend() -> LaMaInferenceBackend:
    return get_default_backend()


def _bad_request(error: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"ok": False, "error": error})


def _server_error(error: str) -> JSONResponse:
    return JSONResponse(status_code=500, content={"ok": False, "error": error})


@app.get("/health")
def health(backend: LaMaInferenceBackend = Depends(get_backend)):
    return {
        "ok": True,
        "service": "ecommerce_text_removal_lama_server",
        "backend": backend.backend_name,
    }


@app.post("/restore")
@app.post("/restore/restore", include_in_schema=False)
async def restore(
    image: UploadFile | None = File(None),
    execution_mask: UploadFile | None = File(None),
    protect_mask: UploadFile | None = File(None),
    scene_type: str = Form(""),
    route: str = Form(""),
    num_candidates: str = Form("1"),
    return_all_candidates: str = Form("0"),
    backend: LaMaInferenceBackend = Depends(get_backend),
):
    del num_candidates
    del return_all_candidates

    if image is None:
        return _bad_request("missing_required_file:image")
    if execution_mask is None:
        return _bad_request("missing_required_file:execution_mask")

    try:
        image_rgb = read_rgb_image_bytes(await image.read())
        execution = read_binary_mask_bytes(await execution_mask.read())
        protect = read_binary_mask_bytes(await protect_mask.read()) if protect_mask is not None else None
    except ValueError as exc:
        return _bad_request(str(exc))

    if execution.shape != image_rgb.shape[:2]:
        return _bad_request("image_execution_mask_size_mismatch")
    if protect is not None and protect.shape != image_rgb.shape[:2]:
        return _bad_request("image_protect_mask_size_mismatch")

    try:
        restored_rgb = backend.restore(
            image_rgb=image_rgb,
            execution_mask=execution,
            protect_mask=protect,
        )
    except LaMaBackendError as exc:
        LOGGER.exception("LaMa backend failed during restore")
        return _server_error(str(exc))
    except Exception as exc:
        LOGGER.exception("Unexpected LaMa restore failure")
        return _server_error(f"unexpected_restore_failure:{exc}")

    if not isinstance(restored_rgb, np.ndarray) or restored_rgb.shape != image_rgb.shape:
        LOGGER.error(
            "LaMa backend returned invalid output: type=%s shape=%s dtype=%s expected_shape=%s",
            type(restored_rgb).__name__,
            getattr(restored_rgb, "shape", "N/A"),
            getattr(restored_rgb, "dtype", "N/A"),
            image_rgb.shape,
        )
        return _server_error("invalid_backend_output")

    return {
        "best_image_hex": encode_png_hex(restored_rgb),
        "meta": {
            "backend": backend.backend_name,
            "used_execution_mask": True,
            "used_protect_mask": bool(protect is not None and backend.uses_protect_mask),
            "scene_type": scene_type,
            "route": route,
        },
    }
