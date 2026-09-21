# Qudurat Extraction Pipeline & Review Dashboard

Extracts Qudurat questions from PDFs using an OpenAI Vision model, has the model **solve and
verify each question itself** before saving it, embeds them, and inserts them into Supabase —
with a Streamlit dashboard for a human to review anything the model wasn't fully confident about.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install Poppler (required by `pdf2image`):
   - Windows: download a Poppler release (e.g. from https://github.com/oschwartz10612/poppler-windows/releases),
     unzip it, and pass its `bin` folder via `--poppler-path`, e.g.
     `--poppler-path "C:\poppler\Library\bin"`.
   - macOS: `brew install poppler`
   - Linux: `apt install poppler-utils`
3. Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`.
4. Run `schema.sql` in the Supabase SQL Editor. It enables `pgvector` and (re)creates the
   `qudurat_quant_questions` table — **note it drops the table first**, so back up any existing
   data you care about before running it.

## Usage — extraction

Process one PDF:
```bash
python extract_pipeline.py --pdf pdfs/algebra.pdf --skill جبر --poppler-path "C:\poppler\Library\bin"
```

Process every PDF in `pdfs/` (skill name = filename stem):
```bash
python extract_pipeline.py --all --pdf-dir pdfs --poppler-path "C:\poppler\Library\bin"
```

Dry run (extract + embed, write to `dry_run_<skill>.jsonl` instead of Supabase):
```bash
python extract_pipeline.py --pdf pdfs/algebra.pdf --skill جبر --dry-run --max-pages 5
```

## Usage — review dashboard

```bash
streamlit run app.py
```

Filter by status / verification / confidence in the sidebar, review each question alongside its
original source image, edit inline if needed, then **Approve & Embed** (re-embeds only if you
changed the text), **Reject**, or **Delete**. Use **Bulk Approve High-Confidence** to approve
every `pending_review` row the model already verified with confidence ≥ 0.95 in one click.

## Self-correction & verification protocol

Before returning a question, the vision model is required to:
1. Independently solve the question itself from the question text it transcribed (shown as
   `solving_work`, not persisted — see `qudurat_pipeline/vision_extractor.py`).
2. Check whether its solved value matches one of the four options it extracted.
3. If not, re-scan the image for a misread sign/digit/denominator (the most common causes: nested
   fractions and small exponents) and re-verify once before giving up.
4. Set `mathematically_verified` and `confidence_score` honestly based on that outcome.

`status` is then computed deterministically in code (not trusted from the model alone):
`approved` only when `mathematically_verified` is true **and** `confidence_score >= 0.95`;
everything else is `pending_review` until a human approves it in the dashboard. `rejected` is only
ever set by a human, via the dashboard.

## Resolution vs. API calls

Pages matching the common Qudurat grid-table template (each question in its own bordered box) are
detected automatically and each question is cropped into its own image before being sent to the
vision model — this gives far better OCR accuracy on small/dense equations than sending the whole
page at once, at the cost of one API call per question. Pages that don't match that template
(diagrams, essay-style layout) fall back to splitting the page into `--strips` overlapping
vertical crops (default 2; set to `1` to send the whole page as one image instead).

## Resuming

Each skill gets a `progress_<skill>.txt` file listing completed page numbers. Re-running the
same command skips pages already processed; a page that fails is left unmarked so it retries
on the next run.

## Adding another skill/section with its own table

`extract_pipeline.py` always writes to `qudurat_quant_questions`. To extract a different
skill/section into its **own** Supabase table (e.g. `المقارنات`) instead of adding a flag to that
script, add a small dedicated entrypoint the same way `extract_pipeline_المقارنات.py` does:

1. Create the table in Supabase with the same columns as `qudurat_quant_questions` in `schema.sql`
   (swap the table name; skip the `ivfflat` index if the embedding model's dimensions exceed 2000).
2. Copy `extract_pipeline_المقارنات.py`, changing `SKILL`, `TABLE`, and `DEFAULT_PDF` at the top.

`qudurat_pipeline.pipeline.process_pdf()` and `supabase_client.insert_rows()` both take an
optional `table` parameter for exactly this — `extract_pipeline.py` itself never needs to change.

## Project layout

- `extract_pipeline.py` — extraction CLI entrypoint (writes to `qudurat_quant_questions`).
- `extract_pipeline_المقارنات.py` — same pipeline, writes to the separate `المقارنات` table.
- `app.py` — Streamlit human-in-the-loop review dashboard.
- `qudurat_pipeline/config.py` — env/settings loading.
- `qudurat_pipeline/schema.py` — Pydantic model, status logic, and the JSON schema handed to the model.
- `qudurat_pipeline/pdf_utils.py` — PDF → in-memory page images + vertical strip splitting (fallback path).
- `qudurat_pipeline/box_detector.py` — detects individual question boxes in the grid-table template.
- `qudurat_pipeline/vision_extractor.py` — OpenAI Vision call, prompt, and self-correction protocol.
- `qudurat_pipeline/embeddings.py` — `text-embedding-3-small` embeddings.
- `qudurat_pipeline/supabase_client.py` — REST insert via `requests`.
- `qudurat_pipeline/checkpoint.py` — per-skill progress file.
- `qudurat_pipeline/pipeline.py` — orchestration, malformed-question filtering, dedup.
- `schema.sql` — Supabase table + pgvector index + status-aware similarity search function.
