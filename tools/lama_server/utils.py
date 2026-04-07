from __future__ import annotations

import binascii
import io

import numpy as np
from PIL import Image


def read_rgb_image_bytes(payload: bytes) -> np.ndarray:
    try:
        return np.array(Image.open(io.BytesIO(payload)).convert("RGB"))
    except Exception as exc:
        raise ValueError(f"invalid_image:{exc}") from exc


def read_binary_mask_bytes(payload: bytes) -> np.ndarray:
    try:
        mask = np.array(Image.open(io.BytesIO(payload)).convert("L"))
    except Exception as exc:
        raise ValueError(f"invalid_mask:{exc}") from exc
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def encode_png_hex(rgb: np.ndarray) -> str:
    bio = io.BytesIO()
    Image.fromarray(rgb.astype("uint8"), "RGB").save(bio, format="PNG")
    return binascii.hexlify(bio.getvalue()).decode("ascii")

