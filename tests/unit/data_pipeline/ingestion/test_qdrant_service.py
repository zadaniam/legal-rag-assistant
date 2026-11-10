from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import PointStruct

from src.data_pipeline.ingestion.qdrant_service import prepare_and_upload_to_qdrant


@pytest.mark.asyncio
async def test_prepare_and_upload_to_qdrant_success(mocker):
    mock_prepared_points = [
        {
            "point_id": "mock-uuid-child-1",
            "parent_id": "mock-uuid-parent-1",
            "text_to_embed": "UU BPJS [BAB I - Pasal 1]: Teks kontens child",
            "text_child_raw": "Teks kontens child",
            "text_parent_utuh": "Pasal 1\nTeks kontens parent utuh satu pasal.",
            "uu_id": "uu_bpjs_2011",
            "bab": "BAB I",
            "pasal": "Pasal 1",
        }
    ]

    mock_embeddings_obj = MagicMock()
    mock_embeddings_obj.values = [0.1] * 3072

    mock_embed_response = MagicMock()
    mock_embed_response.embeddings = [mock_embeddings_obj]

    mock_ai_client = MagicMock()
    mock_ai_client.aio = MagicMock()
    mock_ai_client.aio.models = MagicMock()

    mock_embed_content = AsyncMock(return_value=mock_embed_response)
    mock_ai_client.aio.models.embed_content = mock_embed_content

    mocker.patch("src.data_pipeline.ingestion.qdrant_service.ai_client", mock_ai_client)

    mock_upsert = AsyncMock()
    mocker.patch(
        "src.data_pipeline.ingestion.qdrant_service.async_qdrant_client.upsert",
        mock_upsert,
    )

    await prepare_and_upload_to_qdrant(mock_prepared_points)

    mock_embed_content.assert_called_once_with(
        model="models/gemini-embedding-2",
        contents="UU BPJS [BAB I - Pasal 1]: Teks kontens child",
        config=mocker.ANY,
    )

    mock_upsert.assert_called_once()
    called_kwargs = mock_upsert.call_args.kwargs
    assert "collection_name" in called_kwargs
    assert "points" in called_kwargs

    uploaded_points = called_kwargs["points"]
    assert len(uploaded_points) == 1

    sample_point = uploaded_points[0]
    assert isinstance(sample_point, PointStruct)
    assert sample_point.id == "mock-uuid-child-1"
    assert "text_dense" in sample_point.vector
    assert "text_sparse" in sample_point.vector

    payload = sample_point.payload
    assert payload["text_child"] == "Teks kontens child"
    assert payload["text_parent"] == "Pasal 1\nTeks kontens parent utuh satu pasal."
    assert payload["uu_id"] == "uu_bpjs_2011"
    assert "bagian" not in payload
