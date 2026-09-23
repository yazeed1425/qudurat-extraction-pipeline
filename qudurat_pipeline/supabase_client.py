"""Minimal Supabase REST client for inserting rows via `requests`."""

from __future__ import annotations

import logging
import urllib.parse

import requests

from .config import SUPABASE_TABLE, Settings

logger = logging.getLogger(__name__)


def insert_rows(settings: Settings, rows: list[dict], table: str = SUPABASE_TABLE) -> None:
    """Insert one or more rows into the given Supabase table via the REST API.

    `table` defaults to the standard "الجبر" table, but callers
    (e.g. a separate skill's entrypoint script) can pass a different table name
    — including a non-ASCII one, which is percent-encoded for the request URL.
    """
    if not rows:
        return

    url = f"{settings.supabase_url}/rest/v1/{urllib.parse.quote(table, safe='')}"
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
    logger.info("Inserted %d row(s) into %s", len(rows), table)
