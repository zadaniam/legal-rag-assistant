# src/utils/embeddings.py
import asyncio

from google import genai

from src.config import settings
from src.logger import logger

# Inisialisasi client tunggal
ai_client = genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())


async def embed_single_text(text: str) -> list[float]:
    """Helper untuk mengambil satu vektor dari satu string teks"""
    response = await ai_client.aio.models.embed_content(
        model="models/gemini-embedding-2", contents=text
    )
    return response.embeddings[0].values


async def embed_text(texts: list[str]) -> list[list[float]]:
    """Mengeksekusi pembuatan vektor secara paralel murni menggunakan asyncio.gather"""

    # PERBAIKAN FORENSIK: Proteksi jika data yang dikirim berupa string tunggal (bukan list)
    if isinstance(texts, str):
        texts = [texts]

    try:
        # Menembak API secara bersamaan untuk menghemat waktu latensi network I/O
        tasks = [embed_single_text(t) for t in texts]
        vectors = await asyncio.gather(*tasks)
        return list(vectors)
    except Exception as e:
        logger.error(f"Gagal mengeksekusi batch embedding paralel: {e}")
        raise e
