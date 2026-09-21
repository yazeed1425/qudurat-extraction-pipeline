#!/usr/bin/env python3
"""CLI entrypoint for the Qudurat PDF -> JSON -> embeddings -> Supabase pipeline.

Every extracted question is self-verified by the vision model itself (it solves the
equation and checks its answer against the four options) before being marked
"approved" or "pending_review" — see qudurat_pipeline/vision_extractor.py for the
protocol and app.py for the human-in-the-loop review dashboard.

Examples:
    python extract_pipeline.py --pdf pdfs/algebra.pdf --skill جبر
    python extract_pipeline.py --all --pdf-dir pdfs
    python extract_pipeline.py --pdf pdfs/algebra.pdf --skill جبر --dry-run --max-pages 5
    python extract_pipeline.py --pdf pdfs/algebra.pdf --skill جبر --poppler-path "C:\\poppler\\Library\\bin"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from qudurat_pipeline.config import Settings
from qudurat_pipeline.pipeline import process_pdf

# Windows consoles often default to a legacy codepage (e.g. cp1252) that cannot
# print Arabic text used in skill names and CLI examples; force UTF-8 output.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, help="Path to a single PDF file to process.")
    parser.add_argument("--skill", type=str, help="Skill/category name for this PDF (required with --pdf).")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every PDF in --pdf-dir; each file's skill name is its filename stem.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("pdfs"),
        help="Directory scanned for PDFs when --all is used (default: ./pdfs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and embed as usual but write results to dry_run_<skill>.jsonl instead of Supabase.",
    )
    parser.add_argument("--max-pages", type=int, default=None, help="Limit the number of pages processed per PDF.")
    parser.add_argument(
        "--poppler-path",
        type=str,
        default=None,
        help=r"Path to the Poppler 'bin' folder on Windows, e.g. C:\poppler\Library\bin",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI vision model to use (default: $OPENAI_MODEL or gpt-4o-mini).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI used to rasterize PDF pages (default: 300).")
    parser.add_argument(
        "--strips",
        type=int,
        default=2,
        help="Fallback (non grid-table pages only): split into this many overlapping vertical "
        "crops instead of sending the whole page at once. 1 disables splitting (default: 2).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.all and (not args.pdf or not args.skill):
        print("Error: either --all, or both --pdf and --skill, must be provided.", file=sys.stderr)
        return 1

    settings = Settings.load()
    model = args.model or settings.llm_model

    jobs: list[tuple[Path, str]] = []
    if args.all:
        if not args.pdf_dir.is_dir():
            print(f"Error: --pdf-dir '{args.pdf_dir}' is not a directory.", file=sys.stderr)
            return 1
        pdf_files = sorted(args.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {args.pdf_dir}.", file=sys.stderr)
            return 1
        jobs = [(pdf_path, pdf_path.stem) for pdf_path in pdf_files]
    else:
        if not args.pdf.is_file():
            print(f"Error: PDF file not found: {args.pdf}", file=sys.stderr)
            return 1
        jobs = [(args.pdf, args.skill)]

    for pdf_path, skill in jobs:
        logging.info("Processing %s as skill '%s'", pdf_path, skill)
        process_pdf(
            pdf_path=pdf_path,
            skill=skill,
            settings=settings,
            model=model,
            dry_run=args.dry_run,
            max_pages=args.max_pages,
            poppler_path=args.poppler_path,
            dpi=args.dpi,
            strips=args.strips,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
