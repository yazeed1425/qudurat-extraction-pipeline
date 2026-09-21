"""Embedding generation via whichever embedding model config.py resolved."""

from __future__ import annotations

from openai import OpenAI


def get_embedding(client: OpenAI, text: str, model: str) -> list[float]:
    """Return an embedding vector for the given text using the given model."""
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
