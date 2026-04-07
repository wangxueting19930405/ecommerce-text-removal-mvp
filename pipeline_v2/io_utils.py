from __future__ import annotations

from pathlib import Path
from PIL import Image
import numpy as np
import io
import binascii


def read_rgb(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def write_rgb(path: str | Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype("uint8"), "RGB").save(path)


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype("uint8"), "L").save(path)


def image_hex_from_rgb(rgb: np.ndarray) -> str:
    bio = io.BytesIO()
    Image.fromarray(rgb.astype("uint8"), "RGB").save(bio, format="PNG")
    return binascii.hexlify(bio.getvalue()).decode("ascii")


def rgb_from_image_hex(hex_str: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(binascii.unhexlify(hex_str))).convert("RGB"))
