from __future__ import annotations

import os
from pathlib import Path
import tempfile
import requests
import cv2
import numpy as np
from .io_utils import rgb_from_image_hex, write_rgb, write_mask


def _resolve_external_url(external_url: str | None) -> str | None:
    if external_url:
        return external_url.rstrip('/')
    env = os.getenv('LAMA_EXTERNAL_URL', '').strip()
    return env.rstrip('/') if env else None


def run_lama_like(
    image_rgb: np.ndarray,
    region_mask: np.ndarray,
    protect_mask: np.ndarray,
    external_url: str | None = None,
    require_lama: bool = True,
    allow_fallback: bool = False,
) -> tuple[np.ndarray, dict]:
    resolved = _resolve_external_url(external_url)
    if resolved:
        try:
            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                img_path = td / 'image.png'
                exec_path = td / 'execution_mask.png'
                prot_path = td / 'protect_mask.png'
                write_rgb(img_path, image_rgb)
                write_mask(exec_path, region_mask)
                write_mask(prot_path, protect_mask)
                with open(img_path, 'rb') as fi, open(exec_path, 'rb') as fm, open(prot_path, 'rb') as fp:
                    resp = requests.post(
                        resolved + '/restore',
                        files={
                            'image': ('image.png', fi, 'image/png'),
                            'execution_mask': ('execution_mask.png', fm, 'image/png'),
                            'protect_mask': ('protect_mask.png', fp, 'image/png'),
                        },
                        data={
                            'scene_type': 'panel',
                            'route': 'lama_route',
                            'num_candidates': '1',
                            'return_all_candidates': '0',
                        },
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if 'best_image_hex' in data:
                        return rgb_from_image_hex(data['best_image_hex']), {
                            'method': 'external_lama',
                            'used_external_url': resolved,
                            'backend_ok': True,
                            'meta': data.get('meta', {}),
                        }
                    return image_rgb.copy(), {
                        'method': 'external_lama_invalid_response',
                        'used_external_url': resolved,
                        'backend_ok': False,
                        'error': 'missing_best_image_hex',
                    }
        except Exception as exc:
            if allow_fallback:
                out = cv2.inpaint(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), region_mask, 4, cv2.INPAINT_TELEA)
                return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), {
                    'method': 'lama_failed_fallback_telea',
                    'used_external_url': resolved,
                    'backend_ok': False,
                    'error': str(exc),
                }
            return image_rgb.copy(), {
                'method': 'lama_backend_failed',
                'used_external_url': resolved,
                'backend_ok': False,
                'error': str(exc),
            }

    if require_lama and not allow_fallback:
        return image_rgb.copy(), {
            'method': 'lama_unavailable_reject',
            'used_external_url': None,
            'backend_ok': False,
            'error': 'no_external_url',
        }

    out = cv2.inpaint(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), region_mask, 4, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), {
        'method': 'lama_unavailable_fallback_telea',
        'used_external_url': None,
        'backend_ok': False,
        'error': 'no_external_url',
    }
