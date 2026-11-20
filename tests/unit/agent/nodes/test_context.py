from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.agent.nodes.context import (
    RouterOutputSchema,
    SearchQueryPayload,
    context_manager_node,
)
from src.agent.state import LegalAgentState


@pytest.mark.asyncio
@patch("src.agent.nodes.context.get_chat_model")
async def test_context_manager_chitchat_success(
    mock_get_chat_model: MagicMock,
) -> None:
    mock_structured_llm = MagicMock()
    mock_structured_llm.ainvoke = AsyncMock(
        return_value=RouterOutputSchema(
            is_chitchat=True,
            chitchat_response="Halo! Ada yang bisa saya bantu terkait hukum?",
            search_payloads=[],
        )
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_chat_model.return_value = mock_llm

    mock_state: LegalAgentState = {
        "messages": [HumanMessage(content="Halo selamat pagi AI!")],
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


@pytest.mark.asyncio
@patch("src.agent.nodes.context.get_chat_model")
async def test_context_manager_legal_query_success(
    mock_get_chat_model: MagicMock,
) -> None:
    mock_structured_llm = MagicMock()
    mock_structured_llm.ainvoke = AsyncMock(
        return_value=RouterOutputSchema(
            is_chitchat=False,
            chitchat_response="",
            search_payloads=[
                SearchQueryPayload(
                    query="aturan uang pesangon PHK",
                    target_uu="uu_ketenagakerjaan_2003",
                ),
                SearchQueryPayload(
                    query="hak jaminan pensiun BPJS",
                    target_uu="Semua",
                ),
            ],
        )
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_chat_model.return_value = mock_llm

    mock_state: LegalAgentState = {
        "messages": [
            HumanMessage(
                content=(
                    "Bagaimana perhitungan pesangon dan hak BPJS saya jika kena PHK?"
                ),
            )
        ],
        "search_payloads": [],
        "search_filters": {},
        "retrieved_documents": [],
        "system_status": "clear",
        "final_answer": "",
    }

    node_output = await context_manager_node(mock_state)

    assert node_output["search_filters"]["is_chitchat"] is False
    assert len(node_output["search_payloads"]) == 2
    assert node_output["search_payloads"][0]["target_uu"] == "uu_ketenagakerjaan_2003"
    assert node_output["search_payloads"][1]["query"] == "hak jaminan pensiun BPJS"
