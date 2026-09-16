# tests/unit/test_search.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.nodes.search import legal_search_node
from src.agent.state import LegalAgentState


@pytest.mark.asyncio
# PERBAIKAN 1: Mock fungsi embedding agar tidak menembak API Gemini asli saat testing
@patch("src.agent.nodes.search.embed_text", new_callable=AsyncMock)
# PERBAIKAN 2: Arahkan patch ke async_qdrant_client.query_points (bukan .scroll lagi)
@patch(
    "src.agent.nodes.search.async_qdrant_client.query_points", new_callable=AsyncMock
)
async def test_legal_search_node_multi_uu_success(mock_query_points, mock_embed_text):
    # 1. Siapkan mock data ScoredPoint dari Qdrant
    mock_point = MagicMock()
    mock_point.payload = {
        "uu_id": "UU_CIPTA_KERJA_2023",
        "pasal": "156",
        "text_child": "Pasal 156 Ayat 2: Uang pesangon...",
        "text_parent": "Teks lengkap Pasal 156...",
        "status_keberlakuan": "aktif",
    }

    # Bungkus hasil ke dalam objek tiruan response yang memiliki atribut .points
    mock_response = MagicMock()
    mock_response.points = [mock_point]
    mock_query_points.return_value = mock_response

    # Mock return value untuk fungsi embedding (larik angka tiruan)
    mock_embed_text.return_value = [0.1] * 3072

    # 2. Siapkan Mock State berisi Multi-UU payload hasil keluaran Node 2 kemarin
    mock_state: LegalAgentState = {
        "messages": [],
        "summary_memory": "",
        "search_payloads": [
            {"query": "pesangon di ciptaker", "target_uu": "UU_CIPTA_KERJA_2023"},
            {"query": "aturan lembur", "target_uu": "Semua"},
        ],
        "retrieved_documents": [],
        "system_status": "clear",
        "final_answer": "",
    }

    # 3. PERBAIKAN 3: Jalankan fungsi Node menggunakan kata kunci 'await'
    node_output = await legal_search_node(mock_state)

    # 4. Validasi kebenaran logika paralel
    assert "retrieved_documents" in node_output
    assert len(node_output["retrieved_documents"]) == 1
    assert node_output["retrieved_documents"][0]["pasal"] == "156"
    assert node_output["retrieved_documents"][0]["uu_id"] == "UU_CIPTA_KERJA_2023"

    # Pastikan query points dipanggil 2 kali secara paralel karena ada 2 objek query
    assert mock_query_points.call_count == 2
