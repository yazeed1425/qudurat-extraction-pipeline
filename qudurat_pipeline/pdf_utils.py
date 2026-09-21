"""PDF -> in-memory page image conversion using pdf2image/Poppler."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)


def iter_pdf_page_images(
    pdf_path: Path,
    dpi: int = 200,
    poppler_path: Optional[str] = None,
    max_pages: Optional[int] = None,
) -> Iterator[tuple[int, Image.Image]]:
    """Yield (1-indexed page number, PIL.Image) pairs for a PDF, rendered in memory.

    Pages are rendered one at a time (rather than all up front) so a large PDF
    does not hold every page image in memory simultaneously.
    """
    last_page = max_pages if max_pages else None

    page_number = 1
    while True:
        if last_page is not None and page_number > last_page:
            return

        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            poppler_path=poppler_path,
            first_page=page_number,
            last_page=page_number,
        )
        if not images:
            return

        logger.debug("Rendered page %d of %s at %d DPI", page_number, pdf_path.name, dpi)
        yield page_number, images[0]
        page_number += 1


class ImageStrip:
    """A vertical crop of a page image plus whether each of its edges is a true page edge."""

    __slots__ = ("image", "top_is_page_edge", "bottom_is_page_edge")

    def __init__(self, image: Image.Image, top_is_page_edge: bool, bottom_is_page_edge: bool):
        self.image = image
        self.top_is_page_edge = top_is_page_edge
        self.bottom_is_page_edge = bottom_is_page_edge


def split_image_vertically(
    image: Image.Image, num_strips: int = 1, overlap_frac: float = 0.25
) -> list[ImageStrip]:
    """Split a page image into overlapping vertical strips for higher effective resolution.

    Splitting a tall page into strips means each question box occupies a larger
    fraction of the image the vision model sees, improving OCR fidelity on small
    text (e.g. dense nested-fraction equations) at the cost of extra API calls.
    Consecutive strips overlap generously so every box is fully contained, away
    from any artificial edge, in at least one strip. Each returned ImageStrip
    records whether its top/bottom is the true page edge or an artificial cut,
    so the extraction prompt can correctly apply its edge-exclusion rule only to
    artificial cuts (a box legitimately touching the true top/bottom of the page
    must never be excluded).
    """
    if num_strips <= 1:
        return [ImageStrip(image, top_is_page_edge=True, bottom_is_page_edge=True)]

    strip_height = image.height / num_strips
    overlap = strip_height * overlap_frac

    strips = []
    for i in range(num_strips):
        top = max(0, int(i * strip_height - overlap))
        bottom = min(image.height, int((i + 1) * strip_height + overlap))
        strips.append(
            ImageStrip(
                image.crop((0, top, image.width, bottom)),
                top_is_page_edge=(i == 0),
                bottom_is_page_edge=(i == num_strips - 1),
            )
        )
    return strips
