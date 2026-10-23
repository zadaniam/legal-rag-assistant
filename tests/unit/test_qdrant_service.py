# tests/unit/test_qdrant_service.py
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import PointStruct

from ingestion.qdrant_service import prepare_and_upload_to_qdrant


@pytest.mark.asyncio
async def test_prepare_and_upload_to_qdrant_success(mocker):
    # =====================================================================
    # 1. SETUP: Siapkan data tiruan luaran dari modul splitter
    # =====================================================================
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

    # --- MOCKING GOOGLE GENAI CLIENT ---
    # Membuat tiruan struktur response API dari google-genai SDK terbaru (.embeddings.values)
    mock_embeddings_obj = MagicMock()
    mock_embeddings_obj.values = [
        0.1
    ] * 3072  # Simulasi array dense vector 3072 dimensi

    mock_embed_response = MagicMock()
    mock_embed_response.embeddings = mock_embeddings_obj

    # Lakukan patch / intercept pada inisialisasi genai.Client di dalam modul
    mock_genai_client = MagicMock()
    mock_genai_client.models.embed_content.return_value = mock_embed_response
    mocker.patch(
        "ingestion.qdrant_service.genai.Client", return_value=mock_genai_client
    )

    # --- MOCKING ASYNC QDRANT CLIENT ---
    # Karena connection.py mengekspos objek async, kita mock method .upsert menjadi AsyncMock
    mock_upsert = AsyncMock()
    mocker.patch("ingestion.qdrant_service.async_qdrant_client.upsert", mock_upsert)

    # =====================================================================
    # 2. ACTION: Jalankan fungsi qdrant service yang diuji
    # =====================================================================
    await prepare_and_upload_to_qdrant(mock_prepared_points)

    # =====================================================================
    # 3. ASSERT: Validasi apakah interaksi kode internal sudah benar
    # =====================================================================

    # A. Pastikan API Gemini dipanggil dengan argumen model yang benar (text-embedding-004)
    mock_genai_client.models.embed_content.assert_called_once_with(
        model="models/gemini-embedding-2",
        contents="UU BPJS [BAB I - Pasal 1]: Teks kontens child",
        config=mocker.ANY,  # Menerima konfigurasi types.EmbedContentConfig apa pun
    )

    # B. Pastikan metode database .upsert dipanggil tepat 1 kali
    mock_upsert.assert_called_once()

    # C. Bedah argumen yang dikirim ke fungsi .upsert untuk memeriksa keutuhan payload Qdrant
    called_kwargs = mock_upsert.call_args.kwargs
    assert "collection_name" in called_kwargs
    assert "points" in called_kwargs

    uploaded_points = called_kwargs["points"]
    assert len(uploaded_points) == 1

    # Validasi struktur PointStruct yang terbentuk
    sample_point = uploaded_points[0]
    assert isinstance(sample_point, PointStruct)
    assert sample_point.id == "mock-uuid-child-1"
    assert "text_dense" in sample_point.vector
    assert "text_sparse" in sample_point.vector

    # Pastikan metadata krusial masuk dan field 'bagian' yang usang benar-benar hilang
    payload = sample_point.payload
    assert payload["text_child"] == "Teks kontens child"
    assert payload["text_parent"] == "Pasal 1\nTeks kontens parent utuh satu pasal."
    assert payload["uu_id"] == "uu_bpjs_2011"
    assert "bagian" not in payload
