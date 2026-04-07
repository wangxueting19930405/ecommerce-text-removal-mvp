from __future__ import annotations
from env_bootstrap import bootstrap_env

bootstrap_env()
import os
print("GOOGLE_APPLICATION_CREDENTIALS =", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

import argparse
from pathlib import Path

from pipeline_v2.service import remove_text_file


def main() -> None:
    p = argparse.ArgumentParser(description='Run text removal MVP v2 on one image')
    p.add_argument('--image', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--overlay')
    p.add_argument('--diagnostics')
    p.add_argument('--debug-dir')
    p.add_argument('--selected-groups-json')
    p.add_argument('--disable-google-ocr', action='store_true')
    p.add_argument('--external-url')
    p.add_argument('--allow-lama-fallback', action='store_true')
    p.add_argument('--no-require-lama', action='store_true', help='Allow medium regions to skip LaMa requirement')
    args = p.parse_args()

    remove_text_file(
        image_path=Path(args.image),
        output_path=Path(args.output),
        overlay_path=Path(args.overlay) if args.overlay else None,
        diagnostics_path=Path(args.diagnostics) if args.diagnostics else None,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
        selected_groups_json_path=Path(args.selected_groups_json) if args.selected_groups_json else None,
        prefer_google_ocr=not args.disable_google_ocr,
        external_url=args.external_url,
        require_lama=not args.no_require_lama,
        allow_lama_fallback=args.allow_lama_fallback,
    )
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
