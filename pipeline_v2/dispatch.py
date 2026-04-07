from __future__ import annotations

import numpy as np

from .complexity import compute_local_features, classify_region_complexity
from .hard_gate import local_hard_gate
from .lama_adapter import run_lama_like
from .simple_fill import simple_fill_region


def dispatch_and_gate_region(
    image_rgb: np.ndarray,
    region_id: int,
    bbox: tuple[int, int, int, int],
    region_mask: np.ndarray,
    protect_mask: np.ndarray,
    external_url: str | None = None,
    require_lama: bool = True,
    allow_lama_fallback: bool = False,
) -> tuple[np.ndarray, dict]:
    features = compute_local_features(image_rgb, region_mask, product_mask=protect_mask)
    complexity, complexity_debug = classify_region_complexity(features)

    accepted = False
    method = 'reject'
    candidate_rgb = image_rgb.copy()
    gate_reasons = ['complex_region'] if complexity == 'complex' else []
    gate_metrics: dict = {}
    restore_debug = {'method': 'reject_before_restore'}

    if complexity == 'simple':
        candidate_rgb, restore_debug = simple_fill_region(image_rgb, region_mask)
        method = restore_debug['method']
        accepted, gate_reasons, gate_metrics = local_hard_gate(image_rgb, candidate_rgb, region_mask, protect_mask)
    elif complexity == 'medium':
        candidate_rgb, restore_debug = run_lama_like(
            image_rgb,
            region_mask,
            protect_mask,
            external_url=external_url,
            require_lama=require_lama,
            allow_fallback=allow_lama_fallback,
        )
        method = restore_debug['method']
        if restore_debug.get('backend_ok', True) or allow_lama_fallback:
            accepted, gate_reasons, gate_metrics = local_hard_gate(image_rgb, candidate_rgb, region_mask, protect_mask)
        else:
            accepted = False
            gate_reasons = ['lama_unavailable']
            gate_metrics = {'backend_ok': False}

    decision = {
        'region_id': region_id,
        'bbox': list(map(int, bbox)),
        'complexity': complexity,
        'method': method,
        'accepted': accepted,
        'classification': complexity_debug,
        'restore': restore_debug,
        'hard_gate_reasons': gate_reasons,
        'hard_gate_metrics': gate_metrics,
        'pixels': int((region_mask > 0).sum()),
    }
    return candidate_rgb, decision
