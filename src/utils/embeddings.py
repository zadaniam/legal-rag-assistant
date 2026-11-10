"""Embedding utilities for the Google Gemini text embedding model."""

import asyncio

from google import genai

from src.config import settings
from src.logger import logger

ai_client: genai.Client = genai.Client(
    api_key=settings.GEMINI_API_KEY.get_secret_value()
)


async def embed_single_text(text: str) -> list[float]:
    """Generate an embedding vector for a single text string.

    Uses the Gemini embedding model to convert text into a dense vector.
    """
    response = await ai_client.aio.models.embed_content(
        model="models/gemini-embedding-2", contents=text
    )

    if not response.embeddings:
        raise ValueError("Gemini API returned an empty embedding response.")

    values = response.embeddings[0].values
    if values is None:
        raise ValueError("Embedding response values are None.")

    return values


async def embed_text(texts: list[str]) -> list[list[float]]:
    """Generate embedding vectors for multiple texts in parallel.

    Accepts either a single string or a list of strings. If a single string
    is provided, it is wrapped in a list before processing to maintain
    consistent batch behavior across all callers.
    """
    if isinstance(texts, str):
        texts = [texts]

    try:
        tasks = [embed_single_text(t) for t in texts]
        vectors = await asyncio.gather(*tasks)
        return list(vectors)
    except Exception as e:
        logger.error(f"Failed to execute parallel batch embedding: {e}")
        raise e
