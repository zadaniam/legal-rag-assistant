import pytest

from src.utils.embeddings import embed_text


@pytest.mark.asyncio
async def test_embed_text_batch_success():
    sample_texts = ["Halo Asisten Hukum", "Apa syarat PHK?"]

    vectors = await embed_text(sample_texts)

    assert isinstance(vectors, list)
    assert len(vectors) == 2
    assert len(vectors[0]) == 3072
    assert isinstance(vectors[0][0], float)
