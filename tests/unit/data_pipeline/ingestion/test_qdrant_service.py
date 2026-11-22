from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import PointStruct

from src.data_pipeline.ingestion.qdrant_service import prepare_and_upload_to_qdrant
from src.database.vector_schema import (
    EmbeddingSpec,
    LegalChunkPayload,
    LegalStatus,
    PayloadField,
    PreparedVectorPoint,
    VectorName,
)


@pytest.mark.asyncio
async def test_prepare_and_upload_to_qdrant_success(mocker):
    mock_prepared_points = [
        PreparedVectorPoint(
            point_id="mock-uuid-child-1",
            text_to_embed="UU BPJS [BAB I - Pasal 1]: Teks kontens child",
            payload=LegalChunkPayload(
                text_child="Teks kontens child",
                text_parent="Pasal 1\nTeks kontens parent utuh satu pasal.",
                uu_id="uu_bpjs_2011",
                status_keberlakuan=LegalStatus.BERLAKU.value,
                parent_id="mock-uuid-parent-1",
                bab="BAB I",
                pasal="Pasal 1",
            ),
        )
    ]

    mock_embeddings_obj = MagicMock()
    mock_embeddings_obj.values = [0.1] * EmbeddingSpec.DENSE_DIMENSION

    mock_embed_response = MagicMock()
    mock_embed_response.embeddings = [mock_embeddings_obj]

    mock_ai_client = MagicMock()
    mock_ai_client.aio = MagicMock()
    mock_ai_client.aio.models = MagicMock()

    mock_embed_content = AsyncMock(return_value=mock_embed_response)
    mock_ai_client.aio.models.embed_content = mock_embed_content

    mocker.patch("src.data_pipeline.ingestion.qdrant_service.ai_client", mock_ai_client)

    mock_qdrant = MagicMock()
    mock_qdrant.upsert = AsyncMock()

    await prepare_and_upload_to_qdrant(mock_prepared_points, client=mock_qdrant)

    mock_embed_content.assert_called_once_with(
        model=EmbeddingSpec.DENSE_MODEL,
        contents="UU BPJS [BAB I - Pasal 1]: Teks kontens child",
        config=mocker.ANY,
    )

    mock_qdrant.upsert.assert_called_once()
    called_kwargs = mock_qdrant.upsert.call_args.kwargs
    assert "collection_name" in called_kwargs
    assert "points" in called_kwargs

    uploaded_points = called_kwargs["points"]
    assert len(uploaded_points) == 1

    sample_point = uploaded_points[0]
    assert isinstance(sample_point, PointStruct)
    assert sample_point.id == "mock-uuid-child-1"
    assert VectorName.DENSE.value in sample_point.vector
    assert VectorName.SPARSE.value in sample_point.vector

    payload = sample_point.payload
    assert payload[PayloadField.TEXT_CHILD.value] == "Teks kontens child"
    assert (
        payload[PayloadField.TEXT_PARENT.value]
        == "Pasal 1\nTeks kontens parent utuh satu pasal."
    )
    assert payload[PayloadField.UU_ID.value] == "uu_bpjs_2011"
    assert "bagian" not in payload
