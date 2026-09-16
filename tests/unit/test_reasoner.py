# tests/unit/test_reasoner.py
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.agent.nodes.reasoner import legal_reasoner_node
from src.agent.state import LegalAgentState


# TEST 1: Memastikan rute Chit-Chat dilewatkan langsung (Tidak perlu AsyncMock karena jalurnya deterministik)
@pytest.mark.asyncio
async def test_legal_reasoner_chitchat_pass_through():
    mock_state: LegalAgentState = {
        "messages": [HumanMessage(content="Terima kasih AI!")],
        "search_filters": {
            "is_chitchat": True,
            "chitchat_response": "Sama-sama! Senang bisa membantu Anda.",
        },
        "retrieved_documents": [],
        "final_answer": "",
    }

    # PERBAIKAN: Tambahkan await saat panggil node
    node_output = await legal_reasoner_node(mock_state)
    assert node_output["final_answer"] == "Sama-sama! Senang bisa membantu Anda."


# TEST 2: Memastikan Gemini Pro dipanggil secara asinkronus saat ada dokumen hukum
@pytest.mark.asyncio
# PERBAIKAN: Pindahkan jalur target patch ke submodule '.aio.'
@patch("src.agent.nodes.reasoner.ai_client.aio.models.generate_content")
async def test_legal_reasoner_substance_query_success(mock_generate_content):
    # Gunakan AsyncMock agar objek respons bisa di-await di dalam node
    mock_response = AsyncMock()
    mock_response.text = "Berdasarkan rujukan Anda wajib membayar pesangon [UU No. 6/2023 Pasal 156]. Honor AI."
    mock_generate_content.return_value = mock_response

    mock_state: LegalAgentState = {
        "messages": [HumanMessage(content="Berapa pesangon PHK?")],
        "search_filters": {"is_chitchat": False},
        "retrieved_documents": [
            {
                "nama_regulasi": "UU No. 6/2023 tentang Cipta Kerja",
                "pasal": "156",
                "text_parent": "Isi pasal utuh 156...",
            }
        ],
        "final_answer": "",
    }

    # PERBAIKAN: Jalankan fungsi Node menggunakan kata kunci 'await'
    node_output = await legal_reasoner_node(mock_state)
    assert "Pasal 156" in node_output["final_answer"]
    assert mock_generate_content.call_count == 1
