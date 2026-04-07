from __future__ import annotations

import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw

from pipeline_v2.service import remove_text_file

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def make_comparison(inp: Path, overlay: Path, out: Path, comp: Path):
    i = Image.open(inp).convert('RGB')
    o = Image.open(overlay).convert('RGB')
    r = Image.open(out).convert('RGB')
    w, h = i.size
    canvas = Image.new('RGB', (w * 3, h), 'white')
    canvas.paste(i, (0, 0)); canvas.paste(o, (w, 0)); canvas.paste(r, (w * 2, 0))
    d = ImageDraw.Draw(canvas)
    for idx, label in enumerate(['INPUT', 'OVERLAY', 'OUTPUT']):
        d.rectangle([idx * w, 0, idx * w + 240, 36], fill=(255, 255, 255))
        d.text((10 + idx * w, 10), label, fill=(0, 0, 0))
    canvas.save(comp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='benchmark/dataset')
    p.add_argument('--run-id', default='mvp_v2_run')
    p.add_argument('--disable-google-ocr', action='store_true')
    p.add_argument('--external-url')
    args = p.parse_args()

    dataset = Path(args.dataset)
    out_root = Path('benchmark/results') / args.run_id
    rep_root = Path('benchmark/reports')
    out_root.mkdir(parents=True, exist_ok=True)
    rep_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for cat_dir in sorted([d for d in dataset.iterdir() if d.is_dir()]):
        for img in sorted(cat_dir.iterdir()):
            if img.suffix.lower() not in VALID_EXTS:
                continue
            stem = img.stem
            dst = out_root / cat_dir.name / stem
            dst.mkdir(parents=True, exist_ok=True)
            output = dst / 'output.png'
            overlay = dst / 'overlay.png'
            diag = dst / 'diagnostics.json'
            comp = dst / 'comparison.png'
            remove_text_file(img, output, overlay, diag, prefer_google_ocr=not args.disable_google_ocr, external_url=args.external_url)
            make_comparison(img, overlay, output, comp)
            d = json.loads(diag.read_text(encoding='utf-8'))
            rows.append({
                'category': cat_dir.name,
                'image': img.name,
                'output': str(output),
                'comparison': str(comp),
                'region_count': len(d.get('region_decisions', [])),
            })

    (rep_root / f'{args.run_id}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    md_lines = [f'# Benchmark {args.run_id}', '']
    for row in rows:
        md_lines.append(f"- [{row['category']}] {row['image']} -> {row['comparison']}")
    (rep_root / f'{args.run_id}.md').write_text('\n'.join(md_lines), encoding='utf-8')
    print(f'benchmark complete: run_id={args.run_id}, images={len(rows)}')


if __name__ == '__main__':
    main()
