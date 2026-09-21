"""Upload question-box images to Supabase Storage for questions that depend on a
diagram (has_image=true) — so the diagram survives even where the original PDF
isn't available (a different machine, a RAG consumer with no local file access).
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import urllib.parse

import requests
from PIL import Image

from .config import Settings

logger = logging.getLogger(__name__)

BUCKET = "question-images"
_NON_ASCII_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_path_segment(text: str) -> str:
    """Supabase Storage rejects non-ASCII characters in an object key outright
    (a 400 InvalidKey, independent of URL-encoding), so Arabic skill/file names
    can't be used directly. Stripping them naively would collapse every
    Arabic-named skill to the same generic segment and silently overwrite each
    other's images — so any text containing non-ASCII characters gets a short
    content hash appended, keeping paths ASCII-safe while still unique per
    distinct input string."""
    ascii_part = _NON_ASCII_SAFE.sub("_", text).strip("_")
    if text.isascii():
        return ascii_part or "x"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{ascii_part}-{digest}" if ascii_part else digest


def upload_question_image(
    settings: Settings,
    image: Image.Image,
    skill: str,
    source_pdf: str,
    page: int,
    box_index: int | None,
    suffix: str = "",
) -> str | None:
    """Upload a box (or diagram-crop) image and return its public URL, or None if
    the upload fails (a storage hiccup should never take down the whole
    extraction run — the question is still saved, just without this URL).

    `suffix` (e.g. "_diagram") distinguishes a diagram-only crop from the full
    box image for the same question, so the two don't overwrite each other.
    """
    stem = _safe_path_segment(source_pdf.rsplit(".", 1)[0])
    box_part = f"box{box_index}" if box_index is not None else "full"
    path = f"{_safe_path_segment(skill)}/{stem}/page{page}_{box_part}{suffix}.png"

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    url = f"{settings.supabase_url}/storage/v1/object/{BUCKET}/{urllib.parse.quote(path)}"
    headers = {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
        "Content-Type": "image/png",
        "x-upsert": "true",
    }
    try:
        response = requests.post(url, headers=headers, data=buffer.getvalue(), timeout=30)
        if not response.ok:
            logger.warning("Image upload failed for %s (%d): %s", path, response.status_code, response.text[:200])
            return None
    except requests.RequestException as exc:
        logger.warning("Image upload failed for %s: %s", path, exc)
        return None

    return f"{settings.supabase_url}/storage/v1/object/public/{BUCKET}/{urllib.parse.quote(path)}"
