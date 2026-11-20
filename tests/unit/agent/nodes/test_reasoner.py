from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.nodes.reasoner import legal_reasoner_node
from src.agent.state import LegalAgentState


@pytest.mark.asyncio
async def test_legal_reasoner_chitchat_pass_through() -> None:
    mock_state: LegalAgentState = {
        "messages": [HumanMessage(content="Terima kasih AI!")],
        "search_payloads": [],
        "search_filters": {
            "is_chitchat": True,
            "chitchat_response": "Sama-sama! Senang bisa membantu Anda.",
        },
        "retrieved_documents": [],
        "system_status": "clear",
        "final_answer": "",
    }

    node_output = await legal_reasoner_node(mock_state)
    assert node_output["final_answer"] == "Sama-sama! Senang bisa membantu Anda."


@pytest.mark.asyncio
@patch("src.agent.nodes.reasoner.get_chat_model")
async def test_legal_reasoner_substance_query_success(
    mock_get_chat_model: MagicMock,
) -> None:
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(
            content=(
                "Berdasarkan rujukan Anda wajib membayar pesangon [UU No. 6/2023 "
                "Pasal 156]. Honor AI."
            )
        )
    )
    mock_get_chat_model.return_value = mock_llm

    mock_state: LegalAgentState = {
        "messages": [HumanMessage(content="Berapa pesangon PHK?")],
        "search_payloads": [],
        "search_filters": {"is_chitchat": False},
        "retrieved_documents": [
            {
                "uu_id": "uu_ketenagakerjaan_2003",
                "bab": "BAB XII PEMUTUSAN HUBUNGAN KERJA",
                "pasal": "Pasal 156",
                "text_parent": "Isi pasal utuh 156...",
                "text_child": "Komponen pesangon...",
            }
        ],
        "system_status": "clear",
        "final_answer": "",
    }

    node_output = await legal_reasoner_node(mock_state)
    assert "Pasal 156" in node_output["final_answer"]
    assert mock_llm.ainvoke.call_count == 1
