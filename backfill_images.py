#!/usr/bin/env python3
"""Upload storage images for existing has_image=true rows that don't have one yet
(e.g. rows extracted before qudurat_pipeline/storage.py existed).

Re-renders each row's exact source page/box from the original PDF (same logic
the dashboard uses to display it) and uploads it via upload_question_image(),
then PATCHes image_url onto that row. Safe to re-run — it only touches rows
where image_url is still null.

Usage:
    python backfill_images.py --table الجبر --poppler-path "C:\\poppler\\Library\\bin"
    python backfill_images.py --table المقارنات --pdf-dir pdfs --poppler-path "poppler/Library/bin"
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.parse
from pathlib import Path

import requests
from pdf2image import convert_from_path

from qudurat_pipeline.box_detector import detect_question_boxes
from qudurat_pipeline.config import Settings
from qudurat_pipeline.storage import upload_question_image

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


def fetch_rows_missing_image(settings: Settings, table: str) -> list[dict]:
    url = f"{settings.supabase_url}/rest/v1/{urllib.parse.quote(table, safe='')}"
    headers = {"apikey": settings.supabase_key, "Authorization": f"Bearer {settings.supabase_key}"}
    params = {
        "select": "id,skill,source_page,source_pdf,box_index",
        "has_image": "eq.true",
        "image_url": "is.null",
    }
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def patch_image_url(settings: Settings, table: str, row_id: str, image_url: str) -> None:
    url = f"{settings.supabase_url}/rest/v1/{urllib.parse.quote(table, safe='')}"
    headers = {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    response = requests.patch(url, headers=headers, params={"id": f"eq.{row_id}"}, json={"image_url": image_url}, timeout=30)
    response.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True, help="Supabase table to backfill.")
    parser.add_argument("--pdf-dir", type=Path, default=Path("pdfs"), help="Directory containing the source PDFs.")
    parser.add_argument("--poppler-path", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = Settings.load()
    rows = fetch_rows_missing_image(settings, args.table)
    logger.info("Found %d row(s) in '%s' with has_image=true and no image_url yet", len(rows), args.table)

    # Avoid re-rendering the same page repeatedly when several rows share it.
    page_cache: dict[tuple[str, int], list] = {}
    uploaded = 0
    for row in rows:
        pdf_path = args.pdf_dir / row["source_pdf"]
        if not pdf_path.is_file():
            logger.warning("Skipping %s: PDF not found at %s", row["id"], pdf_path)
            continue

        cache_key = (row["source_pdf"], row["source_page"])
        if cache_key not in page_cache:
            images = convert_from_path(
                str(pdf_path),
                dpi=300,
                poppler_path=args.poppler_path,
                first_page=row["source_page"],
                last_page=row["source_page"],
            )
            if not images:
                logger.warning("Skipping %s: page %d not found in %s", row["id"], row["source_page"], pdf_path)
                page_cache[cache_key] = None
                continue
            page_image = images[0]
            boxes = detect_question_boxes(page_image)
            page_cache[cache_key] = [page_image, boxes]

        cached = page_cache[cache_key]
        if cached is None:
            continue
        page_image, boxes = cached

        box_index = row.get("box_index")
        if box_index is not None and boxes and 0 <= box_index < len(boxes):
            source_image = boxes[box_index]
        else:
            source_image = page_image

        image_url = upload_question_image(
            settings, source_image, row["skill"], row["source_pdf"], row["source_page"], box_index
        )
        if image_url:
            patch_image_url(settings, args.table, row["id"], image_url)
            uploaded += 1
            logger.info("Backfilled %s -> %s", row["id"], image_url)
        else:
            logger.warning("Upload failed for %s, left as-is", row["id"])

    logger.info("Done: %d/%d row(s) backfilled", uploaded, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
