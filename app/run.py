from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .planner import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-first construction estimator")
    parser.add_argument("--input", required=True, help="Input folder containing PDF drawings")
    parser.add_argument("--output", required=True, help="Output folder for results and debug artifacts")
    parser.add_argument("--ocr", choices=["auto", "off", "require"], default="auto", help="OCR policy")
    parser.add_argument("--ai", choices=["off", "auto", "require"], default="auto", help="AI ambiguity-resolution policy")
    parser.add_argument("--ai-model", default=None, help="Optional model name for the AI resolver")
    parser.add_argument("--ai-base-url", default=None, help="Optional OpenAI-compatible base URL for the AI resolver")
    parser.add_argument("--render-dpi", type=int, default=200, help="Raster DPI for rendered page images")
    parser.add_argument("--report-format", choices=["md"], default="md", help="Human-readable report format")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    try:
        artifacts = run_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            render_dpi=args.render_dpi,
            ocr_mode=args.ocr,
            ai_mode=args.ai,
            ai_model=args.ai_model,
            ai_base_url=args.ai_base_url,
            report_format=args.report_format,
        )
    except Exception as exc:  # pragma: no cover - top-level failure path
        print(f"Estimator failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote results to {output_dir / 'results.json'}")
    print(f"Perimeter: {artifacts.results.perimeter_exterior_m:.2f} m")
    print(f"Gross wall area: {artifacts.results.gross_outer_wall_area_m2:.2f} m2")
    print(f"Net cladding area: {artifacts.results.net_cladding_area_m2:.2f} m2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
