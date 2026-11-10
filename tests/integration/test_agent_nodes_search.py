"""Integration tests for the legal search node."""

import pytest

from src.agent.nodes.search import legal_search_node
from src.database.connection import async_qdrant_client

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def verify_qdrant_alive():
    """Fixture otomatis untuk memastikan server Qdrant aktif sebelum pengujian berjalan.
    Hanya melakukan pengecekan kesehatan (health check) tanpa menutup koneksi client
    secara global agar tidak merusak status pool HTTP pada pengujian berikutnya.
    """
    collections_response = await async_qdrant_client.get_collections()
    assert collections_response is not None, (
        "Server Qdrant lokal harus merespons aktif."
    )
    yield


async def test_legal_search_node_success():
    mock_state = {
        "search_payloads": [
            {
                "query": "hak pekerja perempuan melahirkan cuti",
                "target_uu": "uu_ketenagakerjaan_2003",
            }
        ],
        "retrieved_documents": [],
    }

    result = await legal_search_node(mock_state)

    assert isinstance(result, dict), "Output from search node must be a dictionary"
    assert "retrieved_documents" in result, "Output must have key 'retrieved_documents'"

    retrieved_docs = result["retrieved_documents"]
    assert isinstance(retrieved_docs, list), "'retrieved_documents' must be a list"

    assert len(retrieved_docs) > 0, "At least one document must be retrieved"

    first_doc = retrieved_docs[0]
    print(f"\n[Test Log] Top document found: Pasal {first_doc.get('pasal')}")

    assert "uu_id" in first_doc
    assert "pasal" in first_doc
    assert "text_child" in first_doc
    assert "text_parent" in first_doc
    assert first_doc["status_keberlakuan"] == "berlaku"


async def test_legal_search_node_empty_payload():
    mock_state_empty = {"search_payloads": [], "retrieved_documents": []}

    result = await legal_search_node(mock_state_empty)

    assert result == {"retrieved_documents": []}
