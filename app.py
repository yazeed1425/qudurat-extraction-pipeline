#!/usr/bin/env python3
"""Streamlit human-in-the-loop review dashboard for extracted Qudurat questions.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
from openai import OpenAI
from pdf2image import convert_from_path

from qudurat_pipeline.box_detector import detect_question_boxes
from qudurat_pipeline.config import Settings, SUPABASE_TABLE
from qudurat_pipeline.embeddings import get_embedding

# Known tables this dashboard can browse. Add an entry here whenever a new
# dedicated extract_pipeline_<skill>.py script starts writing to its own table
# (see README's "Adding another skill/section with its own table" section).
KNOWN_TABLES = [SUPABASE_TABLE, "المقارنات"]

_LATEX_MARKERS = re.compile(
    r"\\(frac|sqrt|times|div|cdot|left|right|Rightarrow|rightarrow|approx|neq|text)\b|\^\{|_\{|\^-?\d"
)


def _ensure_math_rendered(text: str) -> str:
    """Streamlit/KaTeX only renders LaTeX between $...$. The extraction model is
    told to always wrap math that way but occasionally forgets on a given line —
    rather than showing raw backslash-commands as plain text, wrap any line that
    clearly contains LaTeX commands but no $ at all, line by line so plain-text
    lines (e.g. Arabic prose in a multi-line explanation) are left untouched."""
    if not text:
        return text
    wrapped_lines = []
    for line in text.split("\n"):
        if "$" in line or not _LATEX_MARKERS.search(line):
            wrapped_lines.append(line)
        else:
            wrapped_lines.append(f"${line}$")
    return "\n".join(wrapped_lines)


def _rtl_markdown(text: str, key: str) -> None:
    """Render as plain markdown (so Streamlit's own $...$ -> KaTeX rendering still
    runs), inside a container CSS-targeted to flip to RTL block flow.

    Earlier attempts wrapped the text itself in an HTML <div dir="rtl"> string,
    which requires unsafe_allow_html=True — that bypasses Streamlit's KaTeX
    preprocessing entirely (shows raw "$\\frac{4}{5}$" text) and/or scrambles
    the raw character order before KaTeX ever sees it. Styling Streamlit's own
    *container* after the fact avoids both: the content still goes through
    normal st.markdown() untouched, KaTeX renders the math as its own inline
    unit, and only the finished result's block flow direction changes — which
    is exactly what makes a rendered KaTeX span sit correctly to the LEFT of
    the Arabic sentence that leads it, instead of trailing after it.
    """
    with st.container(key=f"rtlblock_{key}"):
        st.markdown(_ensure_math_rendered(text))


RTL_FIELD_CSS = """
<style>
textarea[aria-label="question_text"],
textarea[aria-label="explanation"],
div[data-testid="stTextInput"] input[aria-label="A"],
div[data-testid="stTextInput"] input[aria-label="B"],
div[data-testid="stTextInput"] input[aria-label="C"],
div[data-testid="stTextInput"] input[aria-label="D"],
div[data-testid="stTextInput"] input[aria-label="skill"],
div[data-testid="stTextInput"] input[aria-label="sub_skill"] {
    text-align: right;
}
div[class*="st-key-rtlblock_"] {
    direction: rtl;
    text-align: right;
}
</style>
"""

st.set_page_config(page_title="Qudurat Review Dashboard", layout="wide")

RENDER_DPI = 300  # must match the DPI extract_pipeline.py used, so box_index still lines up


@st.cache_resource
def get_settings() -> Settings:
    return Settings.load()


@st.cache_resource
def get_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
        default_headers={"Accept-Encoding": "identity"},
    )


def _headers() -> dict:
    settings = get_settings()
    return {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
        "Content-Type": "application/json",
    }


def _table_url(table: str) -> str:
    settings = get_settings()
    return f"{settings.supabase_url}/rest/v1/{urllib.parse.quote(table, safe='')}"


def fetch_questions(
    table: str,
    status_filter: list[str],
    verified_filter: Optional[bool],
    min_confidence: float,
    skill_filter: str,
    has_image_filter: Optional[bool],
    limit: int,
    offset: int,
) -> tuple[list[dict], int]:
    url = _table_url(table)
    params = {
        "select": "*",
        "order": "source_pdf.asc,source_page.asc,box_index.asc",
        "limit": str(limit),
        "offset": str(offset),
        "confidence_score": f"gte.{min_confidence}",
    }
    if status_filter:
        params["status"] = "in.(" + ",".join(status_filter) + ")"
    if verified_filter is not None:
        params["mathematically_verified"] = f"eq.{str(verified_filter).lower()}"
    if skill_filter.strip():
        params["skill"] = f"eq.{skill_filter.strip()}"
    if has_image_filter is not None:
        params["has_image"] = f"eq.{str(has_image_filter).lower()}"

    headers = {**_headers(), "Prefer": "count=exact"}
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    total = 0
    content_range = response.headers.get("content-range", "")
    if "/" in content_range:
        try:
            total = int(content_range.split("/")[-1])
        except ValueError:
            total = 0
    return response.json(), total


def patch_question(table: str, row_id: str, fields: dict) -> None:
    url = _table_url(table)
    response = requests.patch(
        url,
        headers={**_headers(), "Prefer": "return=minimal"},
        params={"id": f"eq.{row_id}"},
        json=fields,
        timeout=30,
    )
    response.raise_for_status()


def delete_question(table: str, row_id: str) -> None:
    url = _table_url(table)
    response = requests.delete(
        url,
        headers={**_headers(), "Prefer": "return=minimal"},
        params={"id": f"eq.{row_id}"},
        timeout=30,
    )
    response.raise_for_status()


def bulk_approve_high_confidence(table: str) -> int:
    """Approve every pending_review row that is verified and confidence >= 0.95.
    Returns the number of rows the server reports as affected (via count=exact)."""
    url = _table_url(table)
    params = {
        "status": "eq.pending_review",
        "mathematically_verified": "eq.true",
        "confidence_score": "gte.0.95",
    }
    response = requests.patch(
        url,
        headers={**_headers(), "Prefer": "return=representation,count=exact"},
        params=params,
        json={"status": "approved"},
        timeout=60,
    )
    response.raise_for_status()
    try:
        return len(response.json())
    except ValueError:
        return 0


@st.cache_data(show_spinner=False, max_entries=64)
def render_source_image(
    pdf_dir: str, source_pdf: str, source_page: int, box_index: Optional[int], poppler_path: str
):
    """Re-render the exact page (and box, if we recorded one) this question came from."""
    pdf_path = Path(pdf_dir) / source_pdf
    if not pdf_path.is_file():
        return None
    images = convert_from_path(
        str(pdf_path),
        dpi=RENDER_DPI,
        poppler_path=poppler_path or None,
        first_page=source_page,
        last_page=source_page,
    )
    if not images:
        return None
    page_image = images[0]
    if box_index is None:
        return page_image
    boxes = detect_question_boxes(page_image)
    if boxes and 0 <= box_index < len(boxes):
        return boxes[box_index]
    return page_image


def render_question_card(row: dict, table: str, pdf_dir: str, poppler_path: str) -> None:
    row_id = row["id"]
    verified = bool(row.get("mathematically_verified"))
    confidence = float(row.get("confidence_score") or 0.0)
    status = row.get("status", "pending_review")

    with st.container(border=True):
        badge = {"approved": "🟢", "rejected": "🔴", "pending_review": "🟡"}.get(status, "⚪")
        header = f"{badge} **{status}** — {row.get('skill','')} / {row.get('sub_skill','')} — page {row.get('source_page')}"
        if not verified or confidence < 0.95:
            header += "  ⚠️ **needs review** (unverified or low confidence)"
        if row.get("has_image"):
            header += "  🖼️ **relies on a diagram** (check the source image, text alone may be incomplete)"
        st.markdown(header)
        st.caption(f"verified={verified}  confidence={confidence:.2f}  id={row_id}")

        col_img, col_form = st.columns([1, 1.4])

        with col_img:
            image_url = row.get("image_url")
            if image_url:
                # Stored at extraction time (has_image=true questions only) —
                # prefer it over a local re-render: it works even if the PDF
                # isn't on this machine, and it's what a remote RAG consumer
                # actually sees too.
                st.image(image_url, use_container_width=True)
                st.caption("🖼️ stored diagram image")
            else:
                try:
                    image = render_source_image(
                        pdf_dir, row.get("source_pdf", ""), row["source_page"], row.get("box_index"), poppler_path
                    )
                except Exception as exc:  # pdf2image / poppler failure
                    image = None
                    st.caption(f"(couldn't render source image: {exc})")
                if image is not None:
                    st.image(image, use_container_width=True)
                else:
                    st.info("Source image unavailable (PDF not found at this path).")

        with col_form:
            q_text = st.text_area(
                "question_text", value=row.get("question_text", ""), key=f"qt_{row_id}", height=90
            )
            _rtl_markdown(q_text, key=f"qt_{row_id}")  # live LaTeX preview (KaTeX), RTL-aware

            options = row.get("options") or {}
            oc1, oc2 = st.columns(2)
            with oc1:
                opt_a = st.text_input("A", value=options.get("A", ""), key=f"a_{row_id}")
                opt_b = st.text_input("B", value=options.get("B", ""), key=f"b_{row_id}")
            with oc2:
                opt_c = st.text_input("C", value=options.get("C", ""), key=f"c_{row_id}")
                opt_d = st.text_input("D", value=options.get("D", ""), key=f"d_{row_id}")
            if any([opt_a, opt_b, opt_c, opt_d]):
                st.markdown(
                    " | ".join(
                        f"**{k}**: {_ensure_math_rendered(v)}"
                        for k, v in [("A", opt_a), ("B", opt_b), ("C", opt_c), ("D", opt_d)]
                    )
                )

            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                answer_options = ["(none)", "A", "B", "C", "D"]
                current_answer = row.get("correct_answer") or "(none)"
                correct_answer = st.selectbox(
                    "correct_answer",
                    answer_options,
                    index=answer_options.index(current_answer) if current_answer in answer_options else 0,
                    key=f"ans_{row_id}",
                )
            with fc2:
                skill = st.text_input("skill", value=row.get("skill", ""), key=f"skill_{row_id}")
            with fc3:
                sub_skill = st.text_input("sub_skill", value=row.get("sub_skill", ""), key=f"subskill_{row_id}")

            with st.expander("explanation"):
                explanation = st.text_area(
                    "explanation", value=row.get("explanation", ""), key=f"exp_{row_id}", height=80
                )
                _rtl_markdown(explanation, key=f"exp_{row_id}")

            b1, b2, b3 = st.columns(3)
            with b1:
                approve_clicked = st.button("✅ Approve & Embed", key=f"approve_{row_id}", use_container_width=True)
            with b2:
                reject_clicked = st.button("🚫 Reject", key=f"reject_{row_id}", use_container_width=True)
            with b3:
                delete_clicked = st.button("🗑️ Delete", key=f"delete_{row_id}", use_container_width=True)

            if approve_clicked:
                new_options = {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d}
                fields = {
                    "question_text": q_text,
                    "options": new_options,
                    "correct_answer": None if correct_answer == "(none)" else correct_answer,
                    "skill": skill,
                    "sub_skill": sub_skill,
                    "explanation": explanation,
                    "status": "approved",
                }
                text_changed = q_text != row.get("question_text", "") or new_options != options
                if text_changed:
                    embed_text = f"{q_text}\n" + " ".join(
                        f"{k}) {v}" for k, v in new_options.items() if v
                    )
                    fields["embedding"] = get_embedding(
                        get_openai_client(), embed_text.strip(), get_settings().embedding_model
                    )
                patch_question(table, row_id, fields)
                st.success("Approved and saved.")
                st.rerun()

            if reject_clicked:
                patch_question(table, row_id, {"status": "rejected"})
                st.warning("Marked as rejected.")
                st.rerun()

            if delete_clicked:
                delete_question(table, row_id)
                st.warning("Deleted.")
                st.rerun()


def main() -> None:
    st.markdown(RTL_FIELD_CSS, unsafe_allow_html=True)
    st.title("📝 Qudurat Review Dashboard")

    try:
        get_settings()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.header("Table")
    table = st.sidebar.selectbox("Table", KNOWN_TABLES, index=0)
    if st.session_state.get("_active_table") != table:
        st.session_state._active_table = table
        st.session_state.offset = 0  # switching tables with a stale offset could skip/hide rows

    st.sidebar.header("Filters")
    status_filter = st.sidebar.multiselect(
        "Status", ["pending_review", "approved", "rejected"], default=["pending_review", "approved"]
    )
    verified_choice = st.sidebar.selectbox("Mathematically verified", ["All", "True", "False"])
    verified_filter = {"All": None, "True": True, "False": False}[verified_choice]
    min_confidence = st.sidebar.slider("Minimum confidence score", 0.0, 1.0, 0.0, 0.01)
    skill_filter = st.sidebar.text_input("Skill (exact match, optional)", value="")
    has_image_choice = st.sidebar.selectbox("Relies on a diagram", ["All", "True", "False"])
    has_image_filter = {"All": None, "True": True, "False": False}[has_image_choice]

    st.sidebar.divider()
    st.sidebar.subheader("Source rendering")
    pdf_dir = st.sidebar.text_input("PDF directory", value="pdfs")
    poppler_path = st.sidebar.text_input("Poppler bin path", value="poppler/Library/bin")

    st.sidebar.divider()
    if st.sidebar.button("⚡ Bulk Approve High-Confidence", use_container_width=True):
        count = bulk_approve_high_confidence(table)
        st.sidebar.success(f"Approved {count} question(s).")
        st.rerun()

    if "offset" not in st.session_state:
        st.session_state.offset = 0
    page_size = 15

    rows, total = fetch_questions(
        table,
        status_filter,
        verified_filter,
        min_confidence,
        skill_filter,
        has_image_filter,
        page_size,
        st.session_state.offset,
    )

    st.caption(f"{total} matching question(s)")

    nav1, nav2, nav3 = st.columns([1, 1, 4])
    with nav1:
        if st.button("⬅️ Previous", disabled=st.session_state.offset == 0):
            st.session_state.offset = max(0, st.session_state.offset - page_size)
            st.rerun()
    with nav2:
        if st.button("Next ➡️", disabled=st.session_state.offset + page_size >= total):
            st.session_state.offset += page_size
            st.rerun()

    if not rows:
        st.info("No questions match the current filters.")
        return

    for row in rows:
        render_question_card(row, table, pdf_dir, poppler_path)


if __name__ == "__main__":
    main()
