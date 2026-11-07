# tests/integration/test_agent_graph.py
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.graph import builder
from src.agent.state import LegalAgentState

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def mock_input_guard_node(_state: LegalAgentState):
    return {"system_status": "clear"}


async def mock_context_manager_node(_state: LegalAgentState):
    return {"search_filters": {"is_chitchat": False}}


async def mock_legal_search_node(_state: LegalAgentState):
    return {
        "search_payloads": [{"query": "pesangon 1 tahun"}],
        "retrieved_documents": [
            {
                "pasal": "Pasal 156",
                "uu_id": "uu_ketenagakerjaan_2003",
                "content": "1 bulan upah",
            }
        ],
    }


async def mock_legal_reasoner_node(_state: LegalAgentState):
    return {
        "final_answer": "Menurut Pasal 156 [UU Ketenagakerjaan], masa kerja kurang dari 1 tahun mendapat 1 bulan upah."
    }


async def mock_output_guard_node(_state: LegalAgentState):
    return _state


@patch("src.agent.graph.AsyncPostgresSaver")
async def test_legal_graph_integration_flow(mock_checkpointer):
    """TEST INTEGRASI: Memastikan struktur 5 Node dan Conditional Edges LangGraph terhubung dengan benar."""

    test_memory = InMemorySaver()

    # Membajak .runnable.afunc menjaga cangkang spec graf asli agar lolos dari validasi kompilasi.
    builder.nodes["input_guard"].runnable.afunc = mock_input_guard_node
    builder.nodes["context_manager"].runnable.afunc = mock_context_manager_node
    builder.nodes["legal_search"].runnable.afunc = mock_legal_search_node
    builder.nodes["legal_reasoner"].runnable.afunc = mock_legal_reasoner_node
    builder.nodes["output_guard"].runnable.afunc = mock_output_guard_node

    integration_graph = builder.compile(checkpointer=test_memory)

    input_state = {
        "messages": [HumanMessage(content="Pertanyaan Hukum Ketenagakerjaan")]
    }
    config = {"configurable": {"thread_id": "test_integration_thread_001"}}

    result = await integration_graph.ainvoke(input_state, config=config)

    assert result["system_status"] == "clear"
    assert result["search_filters"]["is_chitchat"] is False
    assert len(result["search_payloads"]) == 1
    assert len(result["retrieved_documents"]) == 1
    assert "pasal" in result["retrieved_documents"][0]

    assert "[UU" in result["final_answer"]
    assert "1 bulan upah" in result["final_answer"]
