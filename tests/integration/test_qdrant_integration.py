import pytest

from src.agent.nodes.search import legal_search_node
from src.database.connection import async_qdrant_client

# Mengonfigurasi agar semua test di file ini berjalan dalam mode async otomatis
pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def verify_qdrant_alive():
    """
    Fixture otomatis untuk memastikan server Qdrant aktif sebelum pengujian berjalan.
    Hanya melakukan pengecekan kesehatan (health check) tanpa menutup koneksi client
    secara global agar tidak merusak status pool HTTP pada pengujian berikutnya.
    """
    collections_response = await async_qdrant_client.get_collections()
    assert collections_response is not None, (
        "Server Qdrant lokal harus merespons aktif."
    )
    yield


async def test_legal_search_node_success():
    """
    Integration test untuk memastikan node search sukses melakukan pencarian hybrid
    dan mengembalikan dokumen dari mock data yang sudah dimasukkan sebelumnya.
    """
    # 1. Siapkan mock state agen RAG hukum Anda
    mock_state = {
        "search_payloads": [
            {
                "query": "hak pekerja perempuan melahirkan cuti",
                "target_uu": "UU Ketenagakerjaan",
            }
        ],
        "retrieved_documents": [],
    }

    # 2. Eksekusi fungsi node search utama
    result = await legal_search_node(mock_state)

    # 3. Validasi struktur hasil output node
    assert isinstance(result, dict), "Output dari node search harus berupa dictionary"
    assert "retrieved_documents" in result, (
        "Output wajib memiliki key 'retrieved_documents'"
    )

    retrieved_docs = result["retrieved_documents"]
    assert isinstance(retrieved_docs, list), (
        "'retrieved_documents' harus berbentuk list"
    )

    # 4. Validasi keakuratan konten (berdasarkan mock data yang kita ingest sebelumnya)
    assert len(retrieved_docs) > 0, "Harus ada dokumen yang berhasil ditarik"

    first_doc = retrieved_docs[0]
    print(
        f"\n[Test Log] Dokumen teratas yang ditemukan: Pasal {first_doc.get('pasal')}"
    )

    # Cek apakah metadata dasar dari document payload hukum terisi dengan benar
    assert "uu_id" in first_doc
    assert "pasal" in first_doc
    assert "text_child" in first_doc
    assert "text_parent" in first_doc
    assert first_doc["status_keberlakuan"] == "aktif"


async def test_legal_search_node_empty_payload():
    """
    Memastikan node search langsung melewatinya dengan aman jika
    state search_payloads dalam kondisi kosong.
    """
    mock_state_empty = {"search_payloads": [], "retrieved_documents": []}

    result = await legal_search_node(mock_state_empty)

    assert result == {"retrieved_documents": []}
