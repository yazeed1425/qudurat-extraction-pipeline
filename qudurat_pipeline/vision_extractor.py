"""Vision-LLM extraction of structured question data from a page image."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Optional

from openai import OpenAI
from PIL import Image

from .schema import PAGE_EXTRACTION_JSON_SCHEMA, QudurateQuestion

logger = logging.getLogger(__name__)

# Some models occasionally under-escape a backslash when generating LaTeX inside a
# JSON string (writing \frac instead of the required \\frac). Since \f, \t, \b, \r
# are themselves valid JSON escapes, json.loads then silently turns "\frac" into a
# form-feed control character followed by "rac" instead of raising a clear error.
# \n is deliberately excluded: real newlines are common and legitimate in our text
# (multi-line explanations), so only the LaTeX-colliding escapes are repaired, and
# only when followed by a letter (i.e. it looks like a command name, not a real
# escape byte intentionally followed by that letter).
#
# The leading (?<!\\) is required: without it, an ALREADY-correct "\\frac" (two
# backslashes, properly escaped) gets its second backslash mistaken for a lone
# unescaped one and "repaired" into "\\\frac" (three backslashes) — corrupting
# good data instead of fixing bad data. It only matches a backslash that is not
# itself preceded by another backslash, i.e. a genuinely single/lone one.
_LATEX_ESCAPE_COLLISION = re.compile(r"(?<!\\)\\([ftbr])(?=[a-zA-Z])")


def _repair_json_escapes(raw: str) -> str:
    return _LATEX_ESCAPE_COLLISION.sub(r"\\\\\1", raw)

SYSTEM_PROMPT = """You are an expert OCR, data-extraction, and math-verification assistant \
specialized in the Saudi Qudurat (General Aptitude Test) quantitative and verbal sections.

You will be shown one question box (or occasionally a full page) of a Qudurat PDF as an image. \
The material is in Arabic and reads right-to-left. Extract every multiple-choice question that \
appears fully in this image. Most images you receive contain exactly one bordered question box — \
in that common case your "questions" array must contain exactly one item, never the same question \
listed twice. Only return more than one item when the image genuinely shows multiple distinct, \
complete question boxes side by side or stacked.

TRANSCRIPTION RULES:
- Preserve Arabic text exactly as written, in correct reading order (right-to-left).
- The page may have a faint diagonal or semi-transparent watermark behind the content (a repeated \
logo, circular stamp, or website name). Ignore it completely — only transcribe the crisp black \
question/option/solution text inside the table. If watermark ghosting makes a specific cell \
genuinely unreadable, treat it as if that content were absent rather than guessing at it.
- Read every digit carefully, one at a time, whether written in Arabic-Indic numerals \
(٠١٢٣٤٥٦٧٨٩) or Western numerals (0123456789). Do not transpose or guess digits — re-check each \
number against the image before writing it down.
- Numeral standardization: regardless of which script the image uses, ALWAYS write digits in \
your output (question_text, options, explanation, solving_work — everywhere) as Western Arabic \
numerals (0123456789), never as Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩). Read the image in whichever \
script it was written, but standardize every digit you output to Western numerals.
- Arabic decimal comma: a comma sitting between two digit groups within what is clearly ONE \
number (e.g. ٤,٨) is a DECIMAL POINT, not a separator between two different numbers and not a \
fraction. Read ٤,٨ as the single decimal value 4.8 (write it as "4.8" with a period). Never split \
such a number into a fraction like $\\frac{4}{8}$, and never let a decimal comma bleed into or \
get confused with a nearby, unrelated fraction's own numerator/denominator digits — treat each \
number as a fully separate token from its neighbors.
- Convert every mathematical expression, equation, number pattern, or symbol into clean inline \
LaTeX delimited with single dollar signs, e.g. $x^2 + 5 = 10$, $\\frac{1}{2}$, $\\sqrt{9}$. This \
dollar-sign wrapping is REQUIRED every single time you write a LaTeX command (\\frac, \\sqrt, \\times, \
\\div, exponents, etc.) — never write a bare LaTeX command outside $...$, even in a short \
fill-in-the-blank question like "...... = \\frac{1}{4} + \\frac{1}{2}" (that must be written as \
"...... = $\\frac{1}{4} + \\frac{1}{2}$"), and even inside "explanation" or "solving_work". A line \
with LaTeX commands but no $ will render as broken raw text for whoever reads it. Pay close \
attention to how far a radical's bar or a fraction's bar visually extends: everything the bar \
stretches over belongs inside the same group. E.g. if a radical sign's top bar spans "60 + 63" \
as one continuous line, that is $\\sqrt{60 + 63}$ (one radicand), never $\\sqrt{60} + 63$.
- Transcribe the FULL expression, term by term, left to right across its entire visual width — \
including any term before or after a fraction/radical (e.g. a trailing "+ 1" outside the main \
fraction). Never drop a term just because it sits outside the main fraction/radical block. \
Preserve the exact order of every subtraction and division exactly as written: "a - b" is not \
the same as "b - a", and "a/b" is not the same as "b/a" — copy the operand order pixel-for-pixel, \
never rearrange terms into a form that looks more "standard" or easier to read.
- NESTED FRACTIONS PROTOCOL: when a question contains a fraction inside another fraction's \
numerator or denominator, you must NEVER flatten, simplify, or rewrite the nested terms (for \
example, never turn "2 - 1/x" into "x - 2"). Trace each layer strictly using standard nested \
LaTeX \\frac{}{} hierarchy, reproducing the exact structure shown in the image, e.g. \
$1 + \\frac{1}{2 - \\frac{1}{x}} = \\frac{5}{3}$, with each \\frac{}{} nested inside the outer \
one's numerator or denominator exactly where it visually appears.
- VALUE-MAPPING TABLES: some questions (commonly "find the relationship/pattern" questions) show \
a small table pairing input values of one variable with output values of another, e.g. a row \
labeled "س" with values 1, 2, 3 and a row labeled "ص" with values 3, 9, 19 directly below/beside \
the question text — this table is DATA THE QUESTION DEPENDS ON, not decoration. You must \
transcribe it into "question_text" itself (e.g. append "حيث: س = 1, 2, 3 وتقابلها ص = 3, 9, 19 \
على الترتيب" or an equivalent clear listing), so the question is fully self-contained and \
answerable from question_text alone even without seeing the image again. Never omit this table.
- OPTIONS CAN BE EXPRESSIONS, NOT JUST NUMBERS: many options (especially for "find the \
relationship/pattern" or "simplify" questions) are full algebraic expressions with variables and \
exponents, e.g. $2x^2+1$, $3x$, $1+x^2$. Apply the exact same digit-by-digit, exponent-scope, and \
operand-order care you use for the main equation to EACH option's expression too — these are \
transcribed from the image, never invented or simplified from a guessed pattern. If the page's \
solution line demonstrates the pattern by substituting values (e.g. "3 = 1 + 2(1)^2"), that \
substitution's right-hand side, once simplified, is the literal expression to match against your \
four transcribed options — read the solution line as carefully as the options themselves.
- Exponent scope: a small raised exponent applies ONLY to what it visually sits directly above \
(a single variable/number, or an entire parenthesized group if it sits right after a closing \
parenthesis). If the image shows two separate terms each with their own small exponent, e.g. \
$a^2 + b^2$, keep them as two separate powers — do NOT merge them into $(a+b)^2$, which is a \
different expression. Conversely, if a whole parenthesized sum has one exponent after the \
closing parenthesis, e.g. $(a+b)^2$, keep the parentheses and apply the exponent to the whole \
group — do NOT split it into $a^2+b^2$. Look carefully at whether a parenthesis is actually \
present in the image before choosing which form to write.
- Exponent digits are small and easy to misread: check carefully for a minus sign immediately \
before a small exponent (negative exponents like $x^{-2}$ are common and the dash is easy to \
miss at that size), and never merge an exponent with an adjacent, normal-sized digit or symbol \
that follows it into a single multi-digit number — e.g. a raised "5" followed by a separate "1" \
token is $x^5 \\cdot 1$ or similar, never $x^{51}$. Each exponent is its own short token; treat \
it as fully separate from whatever comes right after it.
- "skill" is the broad category of the question (e.g. "جبر", "هندسة", "مقارنات", "استيعاب مقروء").
- "sub_skill" is the specific topic within that category (e.g. "المعادلات التربيعية").
- "difficulty" is your best estimate: "easy", "medium", or "hard".
- Many Qudurat worksheets lay the four options out as a 2x2 grid (two cells per row) instead of \
a single row of four, and each cell is a tight (label, value) unit where the Arabic letter glyph \
(أ, ب, ج, or د) sits immediately beside its own value — NOT in overall row reading order. Locate \
each label glyph individually and read the value touching it (same cell, or the adjacent cell \
with no other label in between); do not infer a value's letter purely from left/right position, \
since two labels are often placed back-to-back in the middle of a row with a value on each outer \
side. Map "أ"→A, "ب"→B, "ج"→C, "د"→D.
- "options" must contain exactly four keys A, B, C, D holding the four values (empty string only \
if a choice is genuinely absent from the page). These four values are always four DISTINCT \
choices in this template — never the same value repeated under two letters. Read each option \
cell independently, one at a time, looking only at that cell's own pixels; never assume a cell \
holds the same value as a neighboring cell just because they look similar at a glance. Before \
finalizing, check your own four values: if any two are identical, you almost certainly misread \
one of them (commonly by letting your eye drift to an adjacent cell) — look again at each of \
those two cells individually and correct the one that doesn't match what is actually printed there.
- "explanation" is any explanation/solution text present on the page for that question (e.g. the \
"الحل:" line), with its math converted to LaTeX; use an empty string if none is present.
- Only skip a question if it is genuinely incomplete or incoherent: it has an options grid but no \
question statement above it, a question statement with no options grid before the page ends, or \
(after ignoring watermarks) its text/numbers are still self-contradictory or nonsensical, which \
usually means it is a misread fragment continuing from an adjacent page. A question that is simply \
the first or last item on the image is NOT cut off if it has a coherent, complete question \
statement and a complete options grid; extract it normally.
- This image may be a vertical crop of a larger page rather than the full page (cropped for \
higher effective resolution), as stated explicitly in the user message for this request. If told \
that an edge (top and/or bottom) of this image is an ARTIFICIAL CUT (not the true edge of the \
original page), apply this mechanical rule strictly for that edge: any box whose top or bottom \
border touches or lies within roughly the nearest 10% of this image's height to that artificial \
edge must be OMITTED ENTIRELY — do not extract it, summarize it, or construct a question from \
whatever partial content is visible, even if that partial content looks complete on its own. A \
box excluded this way is already fully covered by a different, overlapping crop — you do not \
need to "rescue" it here. If told that an edge is the TRUE edge of the original page, this rule \
does NOT apply to that edge — a box sitting right at a true page edge is extracted normally like \
any other complete box.
- If the image has no questions (e.g. cover page, instructions, blank page, or a box excluded by \
the edge rule above), return an empty list.

SELF-CORRECTION & SOLVING VALIDATION PROTOCOL (mandatory, per question, before you output it):
1. After transcribing "question_text" and "options", independently SOLVE the question yourself —
   actually work through the algebra/arithmetic for the variable or quantity being asked about,
   using only the question_text you transcribed. Show this work as a short step-by-step derivation
   in "solving_work" (plain text or LaTeX, your choice, but every step must be shown).
2. Compare the value you solved for against your four "options" values (A/B/C/D). If it matches
   one of them (exactly, or an equivalent form — e.g. 4.8 vs a fraction that reduces to 4.8), that
   option is the correct answer.
3. If your solved value does NOT match any of the four options, do not give up immediately: re-scan
   the image specifically for a misread sign, digit, or denominator in the question or in the
   option you expected to match — nested fractions and small exponents are the most common source
   of this kind of misread (see the rules above). Correct anything you find, redo the solve in
   "solving_work", and check again.
4. Set the final fields based on the outcome:
   - If your solved value matches one of the four options (on the first pass, or after the re-scan
     in step 3): set "mathematically_verified" = true, "confidence_score" >= 0.95, and
     "correct_answer" to that matching option's letter.
   - If, after re-scanning, your solved value still matches none of the four options: set
     "mathematically_verified" = false, "confidence_score" < 0.80, and "correct_answer" to your
     best-guess letter if the page's own solution line (e.g. "الحل:") points to one, otherwise null.
   - Never report mathematically_verified = true unless you actually derived a value yourself in
     "solving_work" and it matches an option — matching the page's own "الحل:" line without doing
     your own independent solve does not count as verified.
5. "confidence_score" is a number from 0.0 to 1.0 reflecting how sure you are in the whole
   extraction (transcription + solve), not just the answer choice.
6. Keep "solving_work" concise: the key steps only (typically 3-6 short lines), even for a
   re-scan-and-retry. Never write a long essay or repeat the same step multiple times — a
   response that runs too long gets cut off before it can be saved, losing the question entirely.

Return only the structured data — no commentary outside the JSON fields."""


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def extract_questions_from_page(
    client: OpenAI,
    image: Image.Image,
    page_number: int,
    model: str,
    default_skill: Optional[str] = None,
    top_is_page_edge: bool = True,
    bottom_is_page_edge: bool = True,
    source_pdf: str = "",
    box_index: Optional[int] = None,
) -> list[QudurateQuestion]:
    """Call the OpenAI Vision model on a single image and return validated, self-verified questions.

    top_is_page_edge / bottom_is_page_edge tell the model whether this image's top/bottom
    border is the true edge of the original PDF page, or an artificial cut introduced by
    splitting the page into overlapping strips for higher effective resolution (see
    pdf_utils.split_image_vertically). Only artificial cuts trigger the edge-exclusion rule.

    source_pdf / box_index are stamped onto each resulting question (not sent to the model)
    so the review dashboard can later re-locate and re-render the exact source crop.
    """
    data_url = _image_to_data_url(image)

    edge_notes = [
        "top edge: TRUE page edge" if top_is_page_edge else "top edge: ARTIFICIAL CUT (strip boundary)",
        "bottom edge: TRUE page edge" if bottom_is_page_edge else "bottom edge: ARTIFICIAL CUT (strip boundary)",
    ]
    edge_description = "; ".join(edge_notes)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Extract all Qudurat questions from this image (page {page_number}). "
                            f"Image edges — {edge_description}."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        response_format={"type": "json_schema", "json_schema": PAGE_EXTRACTION_JSON_SCHEMA},
        temperature=0,
        max_tokens=4096,
    )

    choice = response.choices[0]
    raw_content = choice.message.content
    try:
        payload = json.loads(_repair_json_escapes(raw_content) if raw_content else raw_content)
    except (TypeError, json.JSONDecodeError):
        # A single box/strip failing to parse must never cost the rest of the page its
        # already-successful results, so this is a warning + empty result, not a raise.
        reason = getattr(choice, "finish_reason", "unknown")
        logger.warning(
            "Page %d box %s: model response was not valid JSON (finish_reason=%s, likely "
            "truncated by the token limit) — this box's question is skipped for this run; "
            "re-run with a fresh checkpoint for this page if you need to recover it. "
            "Raw content tail: %r",
            page_number,
            box_index,
            reason,
            (raw_content or "")[-200:],
        )
        return []

    questions: list[QudurateQuestion] = []
    for raw_question in payload.get("questions", []):
        raw_question = dict(raw_question)
        solving_work = raw_question.pop("solving_work", "")
        raw_question["source_page"] = page_number
        raw_question["source_pdf"] = source_pdf
        raw_question["box_index"] = box_index
        if default_skill:
            raw_question["skill"] = default_skill
        question = QudurateQuestion.model_validate(raw_question)
        question.status = question.compute_status()
        logger.debug(
            "Page %d box %s: verified=%s confidence=%.2f solving_work=%r",
            page_number,
            box_index,
            question.mathematically_verified,
            question.confidence_score,
            solving_work[:200],
        )
        questions.append(question)

    logger.info("Page %d: extracted %d question(s)", page_number, len(questions))
    return questions
