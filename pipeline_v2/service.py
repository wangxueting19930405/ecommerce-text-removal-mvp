from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .dispatch import dispatch_and_gate_region
from .io_utils import read_rgb, write_rgb, write_mask
from .ocr import run_ocr
from .protect import build_protect_mask
from .execution_mask import build_execution_mask, iter_regions
from .overlay import draw_overlay
from .types import OCRResult, PipelineResult


def _effective_lama_url(decisions: list[dict], external_url: str | None) -> str | None:
    for d in decisions:
        url = d.get('restore', {}).get('used_external_url')
        if url:
            return url
    if external_url:
        return external_url
    import os
    env = os.getenv('LAMA_EXTERNAL_URL', '').strip()
    return env.rstrip('/') if env else None


def _selected_ids_from_json(path: Path | None, groups, default_ids: set[str]) -> set[str]:
    if not path or not path.exists():
        return set(default_ids)
    data = json.loads(path.read_text(encoding='utf-8'))
    ids = set(data.get('selected_group_ids', []))
    if ids:
        return ids
    by_panel = data.get('selected_group_ids_by_panel')
    if isinstance(by_panel, dict):
        for v in by_panel.values():
            ids.update(v)
    return ids or set(default_ids)


def _build_group_infos(
    image_shape: tuple[int, int],
    ocr_result: OCRResult,
    protect_mask: np.ndarray,
) -> tuple[list[dict], list[str]]:
    group_infos: list[dict] = []
    default_selected_group_ids: list[str] = []
    for g in ocr_result.groups:
        gx1, gy1, gx2, gy2 = g.bbox
        gmask = np.zeros(image_shape, np.uint8)
        for bi in g.box_indices:
            pts = np.array(ocr_result.boxes[bi].polygon, dtype=np.int32)
            cv2.fillPoly(gmask, [pts], 255)
        overlap = float(((gmask > 0) & (protect_mask > 0)).sum()) / max(1, int((gmask > 0).sum()))
        recommendation = 'keep' if overlap >= 0.15 else 'delete'
        if recommendation == 'delete':
            default_selected_group_ids.append(g.group_id)
        group_infos.append({
            'group_id': g.group_id,
            'bbox': [int(gx1), int(gy1), int(gx2), int(gy2)],
            'text': g.text,
            'text_reliable': ocr_result.text_reliable,
            'protect_overlap': overlap,
            'selection_recommendation': recommendation,
            'default_selected': recommendation == 'delete',
        })
    return group_infos, default_selected_group_ids


def _prepare_analysis(
    image_rgb: np.ndarray,
    prefer_google_ocr: bool = True,
) -> dict:
    ocr_result = run_ocr(image_rgb, prefer_google_ocr=prefer_google_ocr)
    protect_mask, protect_debug = build_protect_mask(image_rgb)
    group_infos, default_selected_group_ids = _build_group_infos(image_rgb.shape[:2], ocr_result, protect_mask)
    return {
        'ocr_result': ocr_result,
        'boxes': ocr_result.boxes,
        'group_objects': ocr_result.groups,
        'ocr': ocr_result.debug,
        'ocr_provider': ocr_result.provider,
        'ocr_degraded': ocr_result.degraded,
        'text_reliable': ocr_result.text_reliable,
        'groups': group_infos,
        'group_count': len(group_infos),
        'protect': protect_debug,
        'default_selected_group_ids': default_selected_group_ids,
        'protect_mask': protect_mask,
    }


def analyze_image(
    image_rgb: np.ndarray,
    prefer_google_ocr: bool = True,
) -> dict:
    analysis = _prepare_analysis(
        image_rgb=image_rgb,
        prefer_google_ocr=prefer_google_ocr,
    )
    overlay = draw_overlay(
        image_rgb,
        analysis['group_objects'],
        set(analysis['default_selected_group_ids']),
        analysis['protect_mask'],
        np.zeros(image_rgb.shape[:2], np.uint8),
    )

    return {
        'ocr': analysis['ocr'],
        'ocr_provider': analysis['ocr_provider'],
        'ocr_degraded': analysis['ocr_degraded'],
        'text_reliable': analysis['text_reliable'],
        'groups': analysis['groups'],
        'group_count': analysis['group_count'],
        'protect': analysis['protect'],
        'default_selected_group_ids': analysis['default_selected_group_ids'],
        'protect_mask': analysis['protect_mask'],
        'overlay_rgb': overlay,
    }


def remove_text_image(
    image_rgb: np.ndarray,
    selected_groups_json_path: Path | None = None,
    prefer_google_ocr: bool = True,
    external_url: str | None = None,
    require_lama: bool = True,
    allow_lama_fallback: bool = False,
) -> PipelineResult:
    analysis = _prepare_analysis(
        image_rgb=image_rgb,
        prefer_google_ocr=prefer_google_ocr,
    )
    boxes = analysis['boxes']
    groups = analysis['group_objects']
    protect_mask: np.ndarray = analysis['protect_mask']
    default_ids = set(analysis['default_selected_group_ids'])
    selected_group_ids = _selected_ids_from_json(selected_groups_json_path, groups, default_ids)
    selection_mask, execution_mask, exec_debug = build_execution_mask(
        image_rgb.shape[:2],
        boxes,
        groups,
        selected_group_ids,
        protect_mask,
    )

    output = image_rgb.copy()
    decisions: list[dict] = []
    for region_id, bbox, region_mask in iter_regions(execution_mask):
        candidate_rgb, decision = dispatch_and_gate_region(
            output,
            region_id,
            bbox,
            region_mask,
            protect_mask,
            external_url=external_url,
            require_lama=require_lama,
            allow_lama_fallback=allow_lama_fallback,
        )
        if decision['accepted']:
            output[region_mask > 0] = candidate_rgb[region_mask > 0]
        decisions.append(decision)

    overlay = draw_overlay(image_rgb, groups, selected_group_ids, protect_mask, execution_mask)
    diagnostic_groups = [
        {
            **group_info,
            'selected': group_info['group_id'] in selected_group_ids,
        }
        for group_info in analysis['groups']
    ]
    diagnostics = {
        'ocr_provider': analysis['ocr_provider'],
        'ocr_degraded': analysis['ocr_degraded'],
        'text_reliable': analysis['text_reliable'],
        'ocr': analysis['ocr'],
        'group_count': analysis['group_count'],
        'groups': diagnostic_groups,
        'protect': analysis['protect'],
        'execution': exec_debug,
        'lama': {
            'require_lama': require_lama,
            'allow_lama_fallback': allow_lama_fallback,
            'external_url': _effective_lama_url(decisions, external_url),
            'any_real_lama': any(
                d.get('restore', {}).get('backend_ok') is True
                for d in decisions
            ),
        },
        'region_decisions': decisions,
    }
    return PipelineResult(
        output_rgb=output,
        overlay_rgb=overlay,
        diagnostics=diagnostics,
        selection_mask=selection_mask,
        execution_mask=execution_mask,
        protect_mask=protect_mask,
    )


def remove_text_file(
    image_path: Path,
    output_path: Path,
    overlay_path: Path | None = None,
    diagnostics_path: Path | None = None,
    selected_groups_json_path: Path | None = None,
    prefer_google_ocr: bool = True,
    external_url: str | None = None,
    require_lama: bool = True,
    allow_lama_fallback: bool = False,
    debug_dir: Path | None = None,
) -> None:
    image_rgb = read_rgb(image_path)
    result = remove_text_image(
        image_rgb=image_rgb,
        selected_groups_json_path=selected_groups_json_path,
        prefer_google_ocr=prefer_google_ocr,
        external_url=external_url,
        require_lama=require_lama,
        allow_lama_fallback=allow_lama_fallback,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_rgb(output_path, result.output_rgb)
    if overlay_path:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        write_rgb(overlay_path, result.overlay_rgb)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(result.diagnostics, ensure_ascii=False, indent=2), encoding='utf-8')
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        if result.selection_mask is not None:
            write_mask(debug_dir / 'selection_mask.png', result.selection_mask)
        if result.execution_mask is not None:
            write_mask(debug_dir / 'execution_mask.png', result.execution_mask)
        if result.protect_mask is not None:
            write_mask(debug_dir / 'protect_mask.png', result.protect_mask)
