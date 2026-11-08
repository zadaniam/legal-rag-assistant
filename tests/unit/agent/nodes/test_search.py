# tests/unit/nodes/test_search.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.nodes.search import legal_search_node
from src.agent.state import LegalAgentState


@pytest.mark.anyio
@patch("src.agent.nodes.search.embed_text", new_callable=AsyncMock)
async def test_legal_search_node_multi_query_parallel_success(mock_embed_text, mocker):
    """TEST 1: Memastikan Node Search mampu memproses multi-query secara paralel
    dan menggabungkan hasilnya dengan tepat dari beberapa UU berbeda.
    """
    # 1. Respon untuk Kueri 1 (Pesangon)
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

    # 2. Respon untuk Kueri 2 (BPJS)
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

    # KUNCI PERBAIKAN: Gunakan mocker.patch untuk mengganti metode query_points secara paksa di level target eksekusi node
    mock_query_points = mocker.patch(
        "src.agent.nodes.search.async_qdrant_client.query_points",
        new_callable=AsyncMock,
    )
    mock_query_points.side_effect = [mock_resp_1, mock_resp_2]

    mock_embed_text.return_value = [[0.1] * 3072, [0.2] * 3072]

    mock_state: LegalAgentState = {
        "messages": [],
        "summary_memory": "",
        "search_payloads": [
            {"query": "aturan uang pesangon", "target_uu": "uu_ketenagakerjaan_2003"},
            {"query": "syarat klaim JHT BPJS", "target_uu": "Semua"},
        ],
        "retrieved_documents": [],
        "system_status": "clear",
        "final_answer": "",
    }

    # Eksekusi Node
    node_output = await legal_search_node(mock_state)

    # Validasi Gabungan
    assert "retrieved_documents" in node_output
    assert len(node_output["retrieved_documents"]) == 2
    assert node_output["retrieved_documents"][0]["pasal"] == "Pasal 156"
    assert node_output["retrieved_documents"][1]["pasal"] == "Pasal 37"
    assert mock_query_points.call_count == 2


@pytest.mark.anyio
@patch("src.agent.nodes.search.embed_text", new_callable=AsyncMock)
async def test_legal_search_node_deduplication_handling(mock_embed_text, mocker):
    """TEST 2: Memastikan logika penapisan (deduplication) bekerja membuang dokumen
    jika mendeteksi parent_id yang sama dari dua kueri berbeda.
    """
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

    # KUNCI PERBAIKAN: Gunakan mocker.patch di level target
    mock_query_points = mocker.patch(
        "src.agent.nodes.search.async_qdrant_client.query_points",
        new_callable=AsyncMock,
    )
    mock_query_points.side_effect = [mock_resp_1, mock_resp_2]
    mock_embed_text.return_value = [[0.1] * 3072, [0.1] * 3072]

    mock_state: LegalAgentState = {
        "messages": [],
        "search_payloads": [
            {"query": "upah minimum", "target_uu": "uu_ketenagakerjaan_2003"},
            {"query": "gaji standar", "target_uu": "Semua"},
        ],
        "retrieved_documents": [],
    }

    node_output = await legal_search_node(mock_state)

    assert len(node_output["retrieved_documents"]) == 1
    assert mock_query_points.call_count == 2
