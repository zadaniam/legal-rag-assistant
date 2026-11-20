"""Integration tests for the compiled legal RAG LangGraph workflow with Postgres."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from src.agent.graph import compile_legal_graph
from src.agent.nodes.context import RouterOutputSchema, SearchQueryPayload
from src.agent.nodes.input_guard import InputGuardOutputSchema
from src.agent.nodes.output_guard import OutputAuditorSchema
from src.utils.llm import get_chat_model

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _build_mock_chat_model(
    *,
    guard_output: InputGuardOutputSchema,
    router_output: RouterOutputSchema | None = None,
    reasoner_text: str = "",
    auditor_output: OutputAuditorSchema | None = None,
) -> MagicMock:
    """Create a mock BaseChatModel dispatching structured outputs by schema."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=reasoner_text))

    def _with_structured_output(schema: type[BaseModel]) -> MagicMock:
        structured = MagicMock()
        if schema is InputGuardOutputSchema:
            structured.ainvoke = AsyncMock(return_value=guard_output)
        elif schema is RouterOutputSchema and router_output is not None:
            structured.ainvoke = AsyncMock(return_value=router_output)
        elif schema is OutputAuditorSchema and auditor_output is not None:
            structured.ainvoke = AsyncMock(return_value=auditor_output)
        else:
            raise AssertionError(f"Unexpected schema requested: {schema}")
        return structured

    mock_llm.with_structured_output.side_effect = _with_structured_output
    return mock_llm


@pytest.fixture(autouse=True)
def clear_llm_cache() -> None:
    """Ensure cached chat models do not leak across integration scenarios."""
    get_chat_model.cache_clear()


@patch("src.utils.llm.init_chat_model")
async def test_legal_graph_integration_full_rag_with_postgres(
    mock_init_chat_model: MagicMock,
) -> None:
    """Verify full RAG workflow with real Postgres checkpointer and real Qdrant."""
    mock_init_chat_model.return_value = _build_mock_chat_model(
        guard_output=InputGuardOutputSchema(is_safe=True, reason=""),
        router_output=RouterOutputSchema(
            is_chitchat=False,
            chitchat_response=None,
            search_payloads=[
                SearchQueryPayload(
                    query="hak pekerja perempuan melahirkan cuti",
                    target_uu="uu_ketenagakerjaan_2003",
                )
            ],
        ),
        reasoner_text=(
            "Berdasarkan ketentuan peraturan ketenagakerjaan yang berlaku, "
            "pekerja perempuan berhak memperoleh istirahat melahirkan."
        ),
        auditor_output=OutputAuditorSchema(
            analisis_perbandingan="Substansi sesuai dengan dokumen rujukan.",
            kesimpulan="AMAN",
        ),
    )

    graph = compile_legal_graph()
    config: RunnableConfig = {
        "configurable": {"thread_id": "test_integration_pg_rag_001"}
    }

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Bagaimana hak cuti melahirkan?")]},
        config=config,
    )

    assert result["system_status"] == "clear"
    assert len(result["retrieved_documents"]) > 0
    assert len(result["final_answer"]) > 0

    saved_state = await graph.aget_state(config)
    assert saved_state.values["final_answer"] == result["final_answer"]
