"""Environment and CLI configuration for the Qudurat extraction pipeline.

Both the chat/vision model and the embedding model are called through the
`openai` SDK, but pointed at whichever provider's API key is configured via
its OpenAI-compatible endpoint (e.g. Gemini's) — no separate SDK needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Known embedding model -> vector dimensions, since schema.sql's vector column
# width must match whatever model actually produced the stored embeddings.
EMBEDDING_DIMENSIONS_BY_MODEL = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "gemini-embedding-001": 3072,
}


@dataclass(frozen=True)
class Settings:
    """Secrets and endpoints loaded from the environment / .env file."""

    llm_api_key: str
    llm_model: str
    llm_base_url: str  # "" means the default OpenAI endpoint
    embedding_model: str
    embedding_dimensions: int
    supabase_url: str
    supabase_key: str

    @staticmethod
    def load() -> "Settings":
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        supabase_url = os.environ.get("SUPABASE_URL", "")
        supabase_key = os.environ.get("SUPABASE_KEY", "")

        # Gemini is used whenever its key is present; otherwise fall back to OpenAI.
        if gemini_key:
            llm_api_key = gemini_key
            default_base_url = GEMINI_BASE_URL
            default_model = "gemini-3.6-flash"
            default_embedding_model = "gemini-embedding-001"
        else:
            llm_api_key = openai_key
            default_base_url = ""
            default_model = "gpt-4o-mini"
            default_embedding_model = "text-embedding-3-small"

        llm_model = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or default_model
        llm_base_url = os.environ.get("LLM_BASE_URL", default_base_url)
        embedding_model = os.environ.get("EMBEDDING_MODEL", default_embedding_model)
        embedding_dimensions = EMBEDDING_DIMENSIONS_BY_MODEL.get(embedding_model, 1536)

        missing = [
            name
            for name, value in (
                ("GEMINI_API_KEY or OPENAI_API_KEY", llm_api_key),
                ("SUPABASE_URL", supabase_url),
                ("SUPABASE_KEY", supabase_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Set them in a .env file or the environment."
            )

        return Settings(
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            supabase_url=supabase_url.rstrip("/"),
            supabase_key=supabase_key,
        )


SUPABASE_TABLE = "الجبر"
DEFAULT_DPI = 300
CHECKPOINT_DIR = Path(".")
PDF_DIR = Path("pdfs")
