"""End-to-end orchestration: PDF -> images -> vision extraction -> embeddings -> Supabase."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .box_detector import detect_question_boxes
from .checkpoint import Checkpoint
from .config import Settings
from .embeddings import get_embedding
from .pdf_utils import iter_pdf_page_images, split_image_vertically
from .schema import QudurateQuestion
from .supabase_client import insert_rows
from .vision_extractor import extract_questions_from_page

logger = logging.getLogger(__name__)


def _dedupe_questions(questions: list[QudurateQuestion]) -> list[QudurateQuestion]:
    """Drop duplicate questions extracted from overlapping page strips.

    Two extractions of the same box can have minor OCR differences (a misread
    digit, a swapped A/B label) even though they are the same question, so an
    exact text match is too strict. Instead, key on the *set* of the four
    option values (order-independent) when at least three are non-empty —
    that combination reliably identifies "the same box" even across minor
    OCR variance, without merging genuinely different questions.
    """
    seen: set = set()
    deduped = []
    for question in questions:
        values = tuple(question.options.model_dump().values())
        non_empty = [v for v in values if v.strip()]
        if len(non_empty) >= 3:
            signature = frozenset(values)
        else:
            signature = (question.question_text.strip(), values)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(question)
    return deduped


def _filter_malformed_questions(
    questions: list[QudurateQuestion], page_number: int
) -> list[QudurateQuestion]:
    """Drop structurally broken questions:

    - Too many empty options — a strong signal of a fragment extracted from
      content sliced at a strip boundary rather than a real question, since
      this template always renders all four options filled in.
    - Two options sharing the exact same non-empty value — this template's
      four options are always four distinct choices, so a repeated value
      means the model misread one cell (often copying a neighboring cell's
      value instead of reading its own), not a genuine duplicate answer.
    """
    kept = []
    for question in questions:
        values = list(question.options.model_dump().values())
        non_empty = [v for v in values if v.strip()]
        empty_count = len(values) - len(non_empty)
        has_duplicate = len(non_empty) != len(set(non_empty))

        if empty_count >= 2:
            logger.warning(
                "Page %d: dropping malformed question (%d empty options): %r",
                page_number,
                empty_count,
                question.question_text[:60],
            )
            continue
        if has_duplicate:
            logger.warning(
                "Page %d: dropping malformed question (duplicate option value in %s): %r",
                page_number,
                values,
                question.question_text[:60],
            )
            continue
        kept.append(question)
    return kept


def process_pdf(
    pdf_path: Path,
    skill: str,
    settings: Settings,
    model: str,
    dry_run: bool = False,
    max_pages: Optional[int] = None,
    poppler_path: Optional[str] = None,
    dpi: int = 200,
    strips: int = 1,
) -> None:
    """Process a single PDF end to end, resuming from its checkpoint file if present."""
    # Some local environments ship a Brotli decoder that is incompatible with
    # httpx's streaming decode call; requesting uncompressed responses avoids
    # that code path entirely rather than depending on a specific package.
    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
        default_headers={"Accept-Encoding": "identity"},
    )
    checkpoint = Checkpoint(skill)

    dry_run_output: Optional[Path] = None
    if dry_run:
        dry_run_output = Path(f"dry_run_{skill}.jsonl")
        logger.info("Dry run: extracted questions will be written to %s instead of Supabase", dry_run_output)

    total_questions = 0
    for page_number, image in iter_pdf_page_images(
        pdf_path, dpi=dpi, poppler_path=poppler_path, max_pages=max_pages
    ):
        if checkpoint.is_done(page_number):
            logger.info("Page %d already processed for skill '%s', skipping", page_number, skill)
            continue

        try:
            question_boxes = detect_question_boxes(image)
            questions: list[QudurateQuestion] = []

            if question_boxes is not None:
                # Regular grid-table template: each crop is one complete question,
                # tightly bound with no artificial cut. Dedup is still needed as a
                # safety net — a single box's own API response can occasionally
                # list the same question twice inside its "questions" array.
                logger.debug("Page %d: detected %d question box(es)", page_number, len(question_boxes))
                for box_idx, box_image in enumerate(question_boxes):
                    questions.extend(
                        extract_questions_from_page(
                            client,
                            box_image,
                            page_number,
                            model=model,
                            default_skill=skill,
                            top_is_page_edge=True,
                            bottom_is_page_edge=True,
                            source_pdf=pdf_path.name,
                            box_index=box_idx,
                        )
                    )
                questions = _filter_malformed_questions(questions, page_number)
                questions = _dedupe_questions(questions)
            else:
                # Fallback for pages that don't match the regular grid-table
                # template (diagrams, essay-style layout, etc.): split into
                # overlapping strips instead of sending the whole page at once.
                page_strips = split_image_vertically(image, num_strips=strips)
                for strip in page_strips:
                    questions.extend(
                        extract_questions_from_page(
                            client,
                            strip.image,
                            page_number,
                            model=model,
                            default_skill=skill,
                            top_is_page_edge=strip.top_is_page_edge,
                            bottom_is_page_edge=strip.bottom_is_page_edge,
                            source_pdf=pdf_path.name,
                            box_index=None,
                        )
                    )
                questions = _filter_malformed_questions(questions, page_number)
                if len(page_strips) > 1:
                    questions = _dedupe_questions(questions)

            rows = []
            for question in questions:
                embedding = get_embedding(client, question.embedding_text(), settings.embedding_model)
                rows.append(question.to_supabase_row(embedding))

            if dry_run:
                with dry_run_output.open("a", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            elif rows:
                insert_rows(settings, rows)

            checkpoint.mark_done(page_number)
            total_questions += len(rows)
            approved = sum(1 for q in questions if q.status == "approved")
            logger.info(
                "Page %d: %d question(s) saved (%d auto-approved, %d pending review)",
                page_number,
                len(rows),
                approved,
                len(rows) - approved,
            )

        except Exception:
            logger.exception(
                "Failed to process page %d of %s; leaving unmarked so it retries on next run",
                page_number,
                pdf_path.name,
            )
            continue

    logger.info("Finished %s (skill=%s): %d question(s) processed", pdf_path.name, skill, total_questions)
