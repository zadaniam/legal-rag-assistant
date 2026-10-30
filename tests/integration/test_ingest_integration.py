# tests/integration/test_ingest_integration.py
import asyncio

import pytest
from qdrant_client import models

from ingestion.parser_rag import parse_markdown_to_object
from ingestion.qdrant_service import prepare_and_upload_to_qdrant
from ingestion.splitter import split_laws_into_hierarchical_points
from src.config import settings
from src.database.connection import async_qdrant_client


@pytest.fixture(scope="module")
def event_loop():
    """Memastikan event loop yang digunakan test sama dengan event loop asyncio global"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def integration_mock_markdown(tmp_path):
    """Fixture untuk membuat dokumen hukum fisik .md kecil untuk diuji oleh pipeline"""
    content = (
        "# UNDANG-UNDANG REPUBLIK INDONESIA INTEGRASI TEST\n\n"
        "## BAB I KETENTUAN INTEGRASI\n\n"
        "### Pasal 100\n"
        "Pipa data harus berjalan lancar dari loader parser splitter hingga ke Qdrant.\n\n"
        "### Pasal 101\n"
        "Hybrid search harus memadukan dense vector dan leksikal sparse indeks."
    )
    p = tmp_path / "uu_integration_test.md"
    p.write_text(content, encoding="utf-8")
    return str(p)


@pytest.mark.asyncio
async def test_end_to_end_ingestion_pipeline_success(integration_mock_markdown):
    # =====================================================================
    # 1. SETUP & VALIDASI INFRASTRUKTUR LOKAL (DOCKER)
    # =====================================================================
    # Pastikan testing berjalan di mode development (Docker lokal) agar tidak merusak data cloud produksi
    if settings.ENVIRONMENT == "production":
        pytest.skip(
            "Melewati integration test: Terdeteksi lingkungan PRODUCTION. Berbahaya bagi data cloud asli!"
        )

    target_uu_id = "uu_integration_test_id"
    collection_name = settings.QDRANT_COLLECTION_NAME

    # Cek apakah Docker Qdrant lokal menyala sebelum tes dijalankan
    try:
        await async_qdrant_client.get_collections()
    except Exception:
        pytest.fail(
            "Gagal terhubung ke Qdrant Lokal! Pastikan Docker database sudah Anda nyalakan lewat perintah 'make up' sebelum tes ini dijalankan."
        )

    # =====================================================================
    # 2. ACTION: Jalankan Rantai Komponen Ingestion Layer Secara Berurutan
    # =====================================================================

    # Tahap A: Jalankan Parser Hirarki Hukum Anda
    parsed_laws = parse_markdown_to_object(
        integration_mock_markdown, uu_id=target_uu_id
    )
    assert len(parsed_laws) == 2, "Parser gagal mendeteksi 2 pasal di dalam file .md"

    # Tahap B: Jalankan Splitter Teks Jangkar Murni (UU -> BAB -> Pasal)
    prepared_points = split_laws_into_hierarchical_points(parsed_laws)
    assert len(prepared_points) >= 2, (
        "Splitter gagal memproses data menjadi kepingan child chunks"
    )

    # Tahap C: Jalankan Qdrant Service Asinkron (Menembak Gemini API & Docker Qdrant)
    # CATATAN: Langkah ini memerlukan internet aktif karena memanggil API google-genai untuk kalkulasi vektor
    print(f"-> Mengirim {len(prepared_points)} data integrasi asli ke Docker Qdrant...")
    await prepare_and_upload_to_qdrant(prepared_points)

    # =====================================================================
    # 3. ASSERT: Tarik Data Langsung Dari Docker Qdrant untuk Pembuktian Validitas
    # =====================================================================
    # Beri jeda sangat singkat (1 detik) agar server Qdrant selesai melakukan indexing di background
    await asyncio.sleep(1)

    # Ambil data spesifik dari Qdrant menggunakan Filter uu_id yang baru saja kita indeks
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

    # PEMBUKTIAN 1: Data fisik terbukti masuk ke database dalam bentuk baris data (Points)
    assert len(retrieved_points) > 0, (
        "Gagal menarik data dari Qdrant! Data tidak terbungkus masuk ke DB."
    )

    # PEMBUKTIAN 2: Memeriksa apakah isi vektor dan payload di database cloud/lokal sudah sempurna
    sample_db_point = retrieved_points[0]

    # Vektor dense 3072 dimensi dari Gemini terbukti masuk dengan mulus
    assert "text_dense" in sample_db_point.vector
    assert len(sample_db_point.vector["text_dense"]) == 3072, (
        "Ukuran dimensi vektor di database salah, harus 3072"
    )

    # Indeks kata kunci leksikal leksikal server-side BM25 sukses terbentuk
    assert "text_sparse" in sample_db_point.vector

    # Payload membawa teks induk utuh satu pasal (Rencana satu tabel payload berhasil!)
    payload = sample_db_point.payload
    assert payload["uu_id"] == target_uu_id
    assert "text_parent" in payload
    assert "text_child" in payload
    assert payload["bab"] == "BAB I KETENTUAN INTEGRASI"
    assert payload["status_keberlakuan"] == "berlaku", (
        "Metadata filter 'status_keberlakuan' tidak masuk ke database!"
    )
    assert "bagian" not in payload, (
        "Kesalahan Arsitektur: Field 'bagian' yang sudah usang terdeteksi masuk ke DB!"
    )

    # =====================================================================
    # 4. TEARDOWN: Bersihkan kembali database Docker dari data sampah integrasi ini
    # =====================================================================
    # Menghapus point uji coba berdasarkan ID undang-undang uji coba agar DB Anda tetap bersih
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
        "-> Teardown Sukses: Data integrasi sampah berhasil dibersihkan dari Docker Qdrant."
    )
