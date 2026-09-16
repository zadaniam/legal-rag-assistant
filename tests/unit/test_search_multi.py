# tests/unit/test_search_multi.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.nodes.search import legal_search_node
from src.agent.state import LegalAgentState

# Pastikan unit test tunduk pada cakupan event loop tunggal sesi agar aman
pytestmark = pytest.mark.asyncio(loop_scope="session")


@patch("src.agent.nodes.search.embed_text", new_callable=AsyncMock)
@patch(
    "src.agent.nodes.search.async_qdrant_client.query_points", new_callable=AsyncMock
)
async def test_legal_search_node_multi_query_integration(
    mock_query_points, mock_embed_text
):
    """TEST: Memastikan Node Search mampu memproses multi-query secara paralel dan menggabungkan hasilnya"""

    # 1. Simulasikan fungsi embedding mengembalikan 2 objek vektor (2D list)
    # Kueri 1 mendapatkan vektor berisi angka 0.1, Kueri 2 mendapatkan vektor berisi angka 0.2
    mock_embed_text.return_value = [[0.1] * 3072, [0.2] * 3072]

    # 2. Simulasikan respon Qdrant yang BERBEDA untuk masing-masing kueri paralel
    # Respon untuk Kueri 1 (Pesangon)
    mock_point_pesangon = MagicMock()
    mock_point_pesangon.payload = {
        "pasal": "156",
        "text_parent": "Aturan Uang Pesangon",
    }
    mock_resp_1 = MagicMock()
    mock_resp_1.points = [mock_point_pesangon]

    # Respon untuk Kueri 2 (BPJS)
    mock_point_bpjs = MagicMock()
    mock_point_bpjs.payload = {"pasal": "37", "text_parent": "Aturan Klaim JHT"}
    mock_resp_2 = MagicMock()
    mock_resp_2.points = [mock_point_bpjs]

    # Atur agar query_points mengembalikan mock_resp_1 pada tembakan pertama, dan mock_resp_2 pada tembakan kedua
    mock_query_points.side_effect = [mock_resp_1, mock_resp_2]

    # 3. Siapkan Mock State dengan 2 kueri hasil dekomposisi
    mock_state: LegalAgentState = {
        "messages": [],
        "summary_memory": "",
        "search_payloads": [
            {"query": "aturan uang pesangon", "target_uu": "UU Cipta Kerja"},
            {"query": "syarat klaim JHT BPJS", "target_uu": "Semua"},
        ],
        "retrieved_documents": [],
        "system_status": "clear",
        "final_answer": "",
    }

    # 4. Jalankan fungsi Search Node Async
    node_output = await legal_search_node(mock_state)

    # 5. VALIDASI LOGIKA PARALEL & GABUNGAN
    assert "retrieved_documents" in node_output
    # Hasil akhir harus berisi TEPAT 2 dokumen (Gabungan dari Pekerja 1 dan Pekerja 2)
    assert len(node_output["retrieved_documents"]) == 2

    # Pastikan data dokumen yang digabungkan tidak tertukar susunannya
    assert node_output["retrieved_documents"][0]["pasal"] == "156"
    assert node_output["retrieved_documents"][1]["pasal"] == "37"

    # Bukti mutlak async paralel bekerja: query_points wajib terpanggil tepat 2 kali secara simultan
    assert mock_query_points.call_count == 2
