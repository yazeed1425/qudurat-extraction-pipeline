"""Minimal Supabase REST client for inserting rows via `requests`."""

from __future__ import annotations

import logging

import requests

from .config import SUPABASE_TABLE, Settings

logger = logging.getLogger(__name__)


def insert_rows(settings: Settings, rows: list[dict]) -> None:
    """Insert one or more rows into the qudurat_quant_questions table via the Supabase REST API."""
    if not rows:
        return

    url = f"{settings.supabase_url}/rest/v1/{SUPABASE_TABLE}"
    headers = {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    response = requests.post(url, headers=headers, json=rows, timeout=30)
    if not response.ok:
        raise RuntimeError(
            f"Supabase insert failed ({response.status_code}): {response.text}"
        )
    logger.info("Inserted %d row(s) into %s", len(rows), SUPABASE_TABLE)
