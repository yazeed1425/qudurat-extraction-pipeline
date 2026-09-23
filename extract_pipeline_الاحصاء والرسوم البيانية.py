#!/usr/bin/env python3
"""CLI entrypoint for extracting الاحصاء والرسوم البيانية (statistics & charts)
into its own Supabase table.

Identical pipeline/protocol as extract_pipeline.py (self-verification, box
detection, Gemini vision — see qudurat_pipeline/vision_extractor.py), but a
separate, dedicated script so this section's questions land in their own
"الاحصاء والرسوم البيانية" table instead of الجبر, and so
extract_pipeline.py itself never needs to change to support additional
skills/sections. Each new skill that needs its own table gets a small script
like this one; the shared logic all lives in qudurat_pipeline/.

Examples:
    python "extract_pipeline_الاحصاء والرسوم البيانية.py"
    python "extract_pipeline_الاحصاء والرسوم البيانية.py" --dry-run --max-pages 5
    python "extract_pipeline_الاحصاء والرسوم البيانية.py" --poppler-path "C:\\poppler\\Library\\bin"
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

SKILL = "الاحصاء والرسوم البيانية"
TABLE = "الاحصاء والرسوم البيانية"
DEFAULT_PDF = Path("pdfs") / "الاحصاء والرسوم البيانية.pdf"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pdf", type=Path, default=DEFAULT_PDF, help=f"Path to the PDF (default: {DEFAULT_PDF})."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and embed as usual but write results to dry_run_<skill>.jsonl instead of Supabase.",
    )
    parser.add_argument("--max-pages", type=int, default=None, help="Limit the number of pages processed.")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Stop once at least this many questions have been saved (checked after each page).",
    )
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
        help="Vision model to use (default: $LLM_MODEL / $OPENAI_MODEL from .env).",
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

    if not args.pdf.is_file():
        print(f"Error: PDF file not found: {args.pdf}", file=sys.stderr)
        return 1

    settings = Settings.load()
    model = args.model or settings.llm_model

    logging.info("Processing %s as skill '%s' -> table '%s'", args.pdf, SKILL, TABLE)
    process_pdf(
        pdf_path=args.pdf,
        skill=SKILL,
        settings=settings,
        model=model,
        dry_run=args.dry_run,
        max_pages=args.max_pages,
        max_questions=args.max_questions,
        poppler_path=args.poppler_path,
        dpi=args.dpi,
        strips=args.strips,
        table=TABLE,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
