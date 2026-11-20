from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.legal_retriever import (
    QdrantLegalRetriever,
    get_default_legal_retriever,
)


@pytest.mark.anyio
@patch("src.database.legal_retriever.embed_text", new_callable=AsyncMock)
async def test_qdrant_retriever_multi_query_parallel_success(
    mock_embed_text: AsyncMock,
) -> None:
    mock_point_pesangon = MagicMock()
    mock_point_pesangon.id = 101
    mock_point_pesangon.score = 0.9542
    mock_point_pesangon.payload = {
        "uu_id": "uu_ketenagakerjaan_2003",
        "pasal": "Pasal 156",
        "text_child": "Komponen pesangon...",
        "text_parent": "Aturan Uang Pesangon",
        "status_keberlakuan": "berlaku",
        "parent_id": "parent-uu-ketenagakerjaan-156",
    }
    mock_resp_1 = MagicMock()
    mock_resp_1.points = [mock_point_pesangon]

    mock_point_bpjs = MagicMock()
    mock_point_bpjs.id = 202
    mock_point_bpjs.score = 0.8819
    mock_point_bpjs.payload = {
        "uu_id": "uu_bpjs_2011",
        "pasal": "Pasal 37",
        "text_child": "Klaim JHT...",
        "text_parent": "Aturan Klaim JHT",
        "status_keberlakuan": "berlaku",
        "parent_id": "parent-uu-bpjs-37",
    }
    mock_resp_2 = MagicMock()
    mock_resp_2.points = [mock_point_bpjs]

    mock_client = MagicMock()
    mock_client.query_points = AsyncMock(side_effect=[mock_resp_1, mock_resp_2])
    mock_embed_text.return_value = [[0.1] * 3072, [0.2] * 3072]

    retriever = QdrantLegalRetriever(
        client=mock_client, collection_name="test_collection"
    )
    payloads = [
        {
            "query": "aturan uang pesangon",
            "target_uu": "uu_ketenagakerjaan_2003",
        },
        {
            "query": "syarat klaim JHT BPJS",
            "target_uu": "Semua",
        },
    ]

    results = await retriever.hybrid_search(
        search_payloads=payloads, prefetch_limit=20, fusion_limit=3
    )

    assert len(results) == 2
    assert results[0]["pasal"] == "Pasal 156"
    assert results[1]["pasal"] == "Pasal 37"
    assert mock_client.query_points.call_count == 2

    first_call_kwargs = mock_client.query_points.call_args_list[0].kwargs
    assert first_call_kwargs["limit"] == 3
    assert first_call_kwargs["prefetch"][0].limit == 20
    assert len(first_call_kwargs["prefetch"][0].filter.must) == 2

    second_call_kwargs = mock_client.query_points.call_args_list[1].kwargs
    assert len(second_call_kwargs["prefetch"][0].filter.must) == 1


@pytest.mark.anyio
@patch("src.database.legal_retriever.embed_text", new_callable=AsyncMock)
async def test_qdrant_retriever_deduplicates_by_parent_id(
    mock_embed_text: AsyncMock,
) -> None:
    mock_point_1 = MagicMock()
    mock_point_1.id = 501
    mock_point_1.score = 0.90
    mock_point_1.payload = {
        "parent_id": "parent-uu-duplikat",
        "pasal": "Pasal 90",
        "status_keberlakuan": "berlaku",
    }
    mock_resp_1 = MagicMock()
    mock_resp_1.points = [mock_point_1]

    mock_point_2 = MagicMock()
    mock_point_2.id = 502
    mock_point_2.score = 0.85
    mock_point_2.payload = {
        "parent_id": "parent-uu-duplikat",
        "pasal": "Pasal 90",
        "status_keberlakuan": "berlaku",
    }
    mock_resp_2 = MagicMock()
    mock_resp_2.points = [mock_point_2]

    mock_client = MagicMock()
    mock_client.query_points = AsyncMock(side_effect=[mock_resp_1, mock_resp_2])
    mock_embed_text.return_value = [[0.1] * 3072, [0.1] * 3072]

    retriever = QdrantLegalRetriever(
        client=mock_client, collection_name="test_collection"
    )
    payloads = [
        {"query": "upah minimum", "target_uu": "uu_ketenagakerjaan_2003"},
        {"query": "gaji standar", "target_uu": "Semua"},
    ]

    results = await retriever.hybrid_search(
        search_payloads=payloads, prefetch_limit=20, fusion_limit=3
    )

    assert len(results) == 1
    assert results[0]["parent_id"] == "parent-uu-duplikat"
    assert mock_client.query_points.call_count == 2


@pytest.mark.anyio
@patch("src.database.legal_retriever.embed_text", new_callable=AsyncMock)
async def test_qdrant_retriever_handles_embedding_and_query_errors(
    mock_embed_text: AsyncMock,
) -> None:
    mock_client = MagicMock()
    retriever = QdrantLegalRetriever(
        client=mock_client, collection_name="test_collection"
    )

    assert await retriever.hybrid_search([], prefetch_limit=10, fusion_limit=3) == []

    mock_embed_text.side_effect = RuntimeError("Embedding API timeout")
    res_embed_err = await retriever.hybrid_search(
        [{"query": "cuti tahunan"}], prefetch_limit=10, fusion_limit=3
    )
    assert res_embed_err == []

    mock_embed_text.side_effect = None
    mock_embed_text.return_value = [[0.1] * 3072]
    mock_client.query_points = AsyncMock(side_effect=RuntimeError("Qdrant down"))
    res_qdrant_err = await retriever.hybrid_search(
        [{"query": "cuti tahunan"}], prefetch_limit=10, fusion_limit=3
    )
    assert res_qdrant_err == []


def test_get_default_legal_retriever_raises_when_client_uninitialized() -> None:
    with (
        patch("src.database.legal_retriever.async_qdrant_client", None),
        pytest.raises(RuntimeError, match="async_qdrant_client"),
    ):
        get_default_legal_retriever()
