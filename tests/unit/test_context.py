# tests/unit/test_context_manager.py
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.agent.nodes.context import context_manager_node
from src.agent.state import LegalAgentState


# TEST 1: Memastikan Kasus Basa-basi (Chit-chat) diarahkan dengan benar
@pytest.mark.asyncio
@patch("src.agent.nodes.context.ai_client.aio.models.generate_content")
async def test_context_manager_chitchat_success(mock_generate_content):
    # Setup tiruan respon JSON dari Gemini
    mock_response = AsyncMock()
    mock_response.text = (
        '{"is_chitchat": true, '
        '"chitchat_response": "Halo! Ada yang bisa saya bantu terkait hukum?", '
        '"search_payloads": []}'
    )
    mock_generate_content.return_value = mock_response

    mock_state: LegalAgentState = {
        "messages": [HumanMessage(content="Halo selamat pagi AI!")],
        "summary_memory": "",
        "search_payloads": [],
        "search_filters": {},
        "retrieved_documents": [],
        "system_status": "clear",
        "final_answer": "",
    }

    node_output = await context_manager_node(mock_state)

    assert node_output["search_filters"]["is_chitchat"] is True
    assert "Halo! Ada yang bisa" in node_output["search_filters"]["chitchat_response"]
    assert len(node_output["search_payloads"]) == 0


# TEST 2: Memastikan Kasus Substansi Hukum (RAG) memecah query secara paralel
@pytest.mark.asyncio
@patch("src.agent.nodes.context.ai_client.aio.models.generate_content")
async def test_context_manager_legal_query_success(mock_generate_content):
    mock_response = AsyncMock()
    mock_response.text = (
        '{"is_chitchat": false, '
        '"chitchat_response": "", '
        '"search_payloads": ['
        '  {"query": "aturan uang pesangon PHK", "target_uu": "UU Cipta Kerja"},'
        '  {"query": "hak jaminan pensiun BPJS", "target_uu": "Semua"}'
        "]}"
    )
    mock_generate_content.return_value = mock_response

    mock_state: LegalAgentState = {
        "messages": [
            HumanMessage(
                content="Bagaimana perhitungan pesangon dan hak BPJS saya jika kena PHK?"
            )
        ],
        "summary_memory": "",
        "search_payloads": [],
        "search_filters": {},
        "retrieved_documents": [],
        "system_status": "clear",
        "final_answer": "",
    }

    node_output = await context_manager_node(mock_state)

    assert node_output["search_filters"]["is_chitchat"] is False
    assert len(node_output["search_payloads"]) == 2
    assert node_output["search_payloads"][0]["target_uu"] == "UU Cipta Kerja"
    assert node_output["search_payloads"][1]["query"] == "hak jaminan pensiun BPJS"
