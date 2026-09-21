"""Detect individual question-box crops for grid-table Qudurat worksheet templates.

Some Qudurat PDFs (e.g. the "mo7wsab.com" template used in these sample files) render
every question as an independent bordered table box: a question row, option rows,
then a solution block, each separated from the next by a thin horizontal border
line, with genuine blank margin between one box and the next.

The number of internal dividers per box, and the exact pixel size of a "margin"
gap, are NOT constant across the file — a box with an extra value-mapping table,
or a longer/shorter solution, has different band spacing than a "plain" box, and
different pages even use different margin sizes. So gap *size* alone can't
reliably separate "between boxes" from "within a box" (a within-box gap can be
just as small as another page's margin). Instead this checks structure directly:
within a box, its left/right frame lines are a continuous vertical line running
through every internal gap; between two boxes, the margin is genuinely blank in
every column. So a gap is only treated as a real box boundary when NO column at
all stays dark across it — if any column does (i.e. some part of the box's frame
is still there), the bands on both sides belong to the same box.

When no confident structure is found at all (a different template, a page with
diagrams, a paragraph-style page), detect_question_boxes returns None and the
caller should fall back to whole-page or overlapping-strip extraction instead.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

MIN_BOX_HEIGHT = 80  # px at typical extraction DPI; anything shorter is noise, not a real box
VERTICAL_LINE_COVERAGE = 0.85  # fraction of a gap's height a column must stay dark to count as "still framed"


def _find_thin_horizontal_bands(
    image: Image.Image,
    dark_threshold: int = 150,
    dark_frac_threshold: float = 0.5,
    max_band_thickness: int = 15,
) -> list[tuple[int, int]]:
    """Find thin, mostly-dark horizontal bands (table border lines) in a page image.

    A band is a run of consecutive rows that are each mostly dark pixels across
    the row's width. Thick dark regions (banners, footer graphics, watermark
    arcs) are excluded via max_band_thickness so they never get mistaken for a
    table border line, which this template renders as a consistently thin rule.
    """
    gray = np.array(image.convert("L"))
    dark_frac = (gray < dark_threshold).mean(axis=1)
    dark_rows = np.where(dark_frac > dark_frac_threshold)[0]

    bands: list[tuple[int, int]] = []
    if len(dark_rows) == 0:
        return bands

    start = prev = int(dark_rows[0])
    for y in dark_rows[1:]:
        y = int(y)
        if y - prev > 3:
            if prev - start <= max_band_thickness:
                bands.append((start, prev))
            start = y
        prev = y
    if prev - start <= max_band_thickness:
        bands.append((start, prev))
    return bands


def _is_still_framed(gray: np.ndarray, y0: int, y1: int) -> bool:
    """True if some column stays dark across the whole [y0, y1) gap — i.e. a
    box's left/right border is still running through here, so this gap is
    inside a box, not the blank margin between two boxes."""
    if y1 - y0 < 3:
        return True
    col_dark_frac = (gray[y0:y1] < 150).mean(axis=0)
    return bool(col_dark_frac.max() > VERTICAL_LINE_COVERAGE)


def detect_question_boxes(image: Image.Image, padding: int = 6) -> list[Image.Image] | None:
    """Return one tightly-cropped image per question box, or None if this page
    doesn't match the expected grid-table structure (caller should fall back to
    another extraction strategy for this page).
    """
    bands = _find_thin_horizontal_bands(image)
    if len(bands) < 4:
        return None

    gray = np.array(image.convert("L"))
    groups: list[list[tuple[int, int]]] = [[bands[0]]]
    for i in range(len(bands) - 1):
        y0, y1 = bands[i][1], bands[i + 1][0]
        if _is_still_framed(gray, y0, y1):
            groups[-1].append(bands[i + 1])
        else:
            groups.append([bands[i + 1]])

    boxes = []
    for group in groups:
        if len(group) < 2:
            continue  # a lone band with no partner isn't a real box; skip it
        top = max(0, group[0][0] - padding)
        bottom = min(image.height, group[-1][1] + padding)
        if bottom - top < MIN_BOX_HEIGHT:
            continue
        boxes.append(image.crop((0, top, image.width, bottom)))

    return boxes or None
