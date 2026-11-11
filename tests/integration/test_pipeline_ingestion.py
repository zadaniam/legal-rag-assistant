"""Integration test for the data ingestion pipeline."""

import asyncio

import pytest
from qdrant_client import models

from src.config import settings
from src.data_pipeline.ingestion.parser_rag import parse_markdown_to_object
from src.data_pipeline.ingestion.qdrant_service import prepare_and_upload_to_qdrant
from src.data_pipeline.ingestion.splitter import split_laws_into_hierarchical_points
from src.database.connection import async_qdrant_client


@pytest.fixture(scope="module")
def event_loop():
    """Ensures the test uses the same event loop as the global asyncio loop."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def integration_mock_markdown(tmp_path):
    """Creates a small legal .md document for pipeline testing."""
    content = (
        "# UNDANG-UNDANG REPUBLIK INDONESIA INTEGRASI TEST\n\n"
        "## BAB I KETENTUAN INTEGRASI\n\n"
        "### Pasal 100\n"
        "Pipa data harus berjalan lancar dari loader parser splitter hingga "
        "ke Qdrant.\n\n"
        "### Pasal 101\n"
        "Hybrid search harus memadukan dense vector dan leksikal sparse indeks."
    )
    p = tmp_path / "uu_integration_test.md"
    p.write_text(content, encoding="utf-8")
    return str(p)


@pytest.mark.asyncio
async def test_end_to_end_ingestion_pipeline_success(integration_mock_markdown):
    # Skip in production to avoid touching real cloud data
    if settings.ENVIRONMENT == "production":
        pytest.skip(
            "Skipping integration test: PRODUCTION environment detected. "
            "Dangerous for real cloud data!"
        )

    target_uu_id = "uu_integration_test_id"
    collection_name = settings.QDRANT_COLLECTION_NAME

    # Verify local Docker Qdrant is running before tests
    try:
        await async_qdrant_client.get_collections()
    except Exception:
        pytest.fail(
            "Failed to connect to local Qdrant! Start Docker database with "
            "'make up' before running this test."
        )

    parsed_laws = parse_markdown_to_object(
        integration_mock_markdown, uu_id=target_uu_id
    )
    assert len(parsed_laws) == 2, "Parser failed to detect 2 pasal in the .md file"

    prepared_points = split_laws_into_hierarchical_points(parsed_laws)
    assert len(prepared_points) >= 2, (
        "Splitter failed to process data into child chunks"
    )

    # This step requires internet access for google-genai vector calculation
    print(
        f"-> Sending {len(prepared_points)} real integration data to Docker Qdrant..."
    )
    await prepare_and_upload_to_qdrant(prepared_points)

    # Wait briefly for Qdrant background indexing
    await asyncio.sleep(1)

    search_response = await async_qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="uu_id", match=models.MatchValue(value=target_uu_id)
                )
            ]
        ),
        limit=10,
        with_payload=True,
        with_vectors=True,
    )

    retrieved_points, _ = search_response

    assert len(retrieved_points) > 0, (
        "Failed to retrieve data from Qdrant! Data not wrapped into DB."
    )

    sample_db_point = retrieved_points[0]

    assert "text_dense" in sample_db_point.vector
    assert len(sample_db_point.vector["text_dense"]) == 3072, (
        "Database vector dimension is wrong, must be 3072"
    )

    assert "text_sparse" in sample_db_point.vector

    payload = sample_db_point.payload
    assert payload["uu_id"] == target_uu_id
    assert "text_parent" in payload
    assert "text_child" in payload
    assert payload["bab"] == "BAB I KETENTUAN INTEGRASI"
    assert payload["status_keberlakuan"] == "berlaku", (
        "Metadata filter 'status_keberlakuan' not inserted into database!"
    )
    assert "bagian" not in payload, (
        "Architecture error: Deprecated field 'bagian' detected in DB!"
    )

    # Clean up test data from Docker Qdrant
    await async_qdrant_client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="uu_id", match=models.MatchValue(value=target_uu_id)
                    )
                ]
            )
        ),
    )
    print(
        "-> Teardown success: Integration test trash data cleaned from Docker Qdrant."
    )
