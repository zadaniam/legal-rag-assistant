# tests/utils/test_embeddings.py
import pytest

from src.utils.embeddings import embed_text


@pytest.mark.asyncio
async def test_embed_text_batch_success():
    # Setup data uji coba ringan
    sample_texts = ["Halo Asisten Hukum", "Apa syarat PHK?"]

    # Eksekusi fungsi async
    vectors = await embed_text(sample_texts)

    # Validasi Hasil
    assert isinstance(vectors, list)
    assert len(vectors) == 2  # Harus menghasilkan 2 vektor sesuai input
    assert len(vectors[0]) == 3072  # Dimensi model gemini-embedding-2 harus pas 3072
    assert isinstance(vectors[0][0], float)
