from __future__ import annotations
from env_bootstrap import bootstrap_env

bootstrap_env()

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline_v2.io_utils import read_rgb, write_rgb, write_mask
from pipeline_v2.service import analyze_image, remove_text_file

BASE_DIR = Path(__file__).resolve().parent
WEB_OUTPUTS = BASE_DIR / 'web_outputs'
WEB_OUTPUTS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='ecommerce_text_removal_system_v2_mvp')
app.mount('/artifacts', StaticFiles(directory=str(WEB_OUTPUTS)), name='artifacts')


@app.get('/health')
def health():
    return {'ok': True, 'service': 'ecommerce_text_removal_system_v2_mvp'}


def _job_dir(job_id: str) -> Path:
    d = WEB_OUTPUTS / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.post('/analyze')
async def analyze(
    image: UploadFile = File(...),
    disable_google_ocr: str = Form('0'),
):
    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    image_path = job_dir / (image.filename or 'image.png')
    image_path.write_bytes(await image.read())

    image_rgb = read_rgb(image_path)
    result = analyze_image(
        image_rgb=image_rgb,
        prefer_google_ocr=(disable_google_ocr != '1'),
    )

    overlay_path = job_dir / 'analyze_overlay.png'
    protect_mask_path = job_dir / 'analyze_protect_mask.png'
    write_rgb(overlay_path, result['overlay_rgb'])
    write_mask(protect_mask_path, result['protect_mask'])

    payload = {
        'job_id': job_id,
        'image_url': f'/artifacts/{job_id}/{image_path.name}',
        'analyze_overlay_url': f'/artifacts/{job_id}/{overlay_path.name}',
        'protect_mask_url': f'/artifacts/{job_id}/{protect_mask_path.name}',
        'ocr_provider': result['ocr_provider'],
        'ocr_degraded': result['ocr_degraded'],
        'text_reliable': result['text_reliable'],
        'ocr': result['ocr'],
        'groups': result['groups'],
        'group_count': result['group_count'],
        'protect': result['protect'],
        'default_selected_group_ids': result['default_selected_group_ids'],
    }
    return JSONResponse(payload)


@app.post('/remove-text')
async def remove_text(
    image: UploadFile = File(...),
    selected_groups_json: str = Form('{}'),
    external_url: str = Form(''),
    disable_google_ocr: str = Form('0'),
    require_lama: str = Form('1'),
    allow_lama_fallback: str = Form('0'),
):
    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    image_path = job_dir / (image.filename or 'image.png')
    output_path = job_dir / 'output.png'
    overlay_path = job_dir / 'overlay.png'
    diagnostics_path = job_dir / 'diagnostics.json'
    selected_path = job_dir / 'selected_groups.json'
    debug_dir = job_dir / 'debug'

    image_path.write_bytes(await image.read())
    selected_path.write_text(selected_groups_json, encoding='utf-8')

    remove_text_file(
        image_path=image_path,
        output_path=output_path,
        overlay_path=overlay_path,
        diagnostics_path=diagnostics_path,
        debug_dir=debug_dir,
        selected_groups_json_path=selected_path,
        prefer_google_ocr=(disable_google_ocr != '1'),
        external_url=external_url or None,
        require_lama=(require_lama != '0'),
        allow_lama_fallback=(allow_lama_fallback == '1'),
    )
    diagnostics = json.loads(diagnostics_path.read_text(encoding='utf-8'))

    payload = {
        'job_id': job_id,
        'image_url': f'/artifacts/{job_id}/{image_path.name}',
        'output_url': f'/artifacts/{job_id}/{output_path.name}',
        'overlay_url': f'/artifacts/{job_id}/{overlay_path.name}',
        'diagnostics_url': f'/artifacts/{job_id}/{diagnostics_path.name}',
        'selection_mask_url': f'/artifacts/{job_id}/debug/selection_mask.png',
        'execution_mask_url': f'/artifacts/{job_id}/debug/execution_mask.png',
        'protect_mask_url': f'/artifacts/{job_id}/debug/protect_mask.png',
        'ocr_provider': diagnostics.get('ocr_provider'),
        'ocr_degraded': diagnostics.get('ocr_degraded'),
        'text_reliable': diagnostics.get('text_reliable'),
        'protect': diagnostics.get('protect'),
        'diagnostics': diagnostics,
    }
    return JSONResponse(payload)
