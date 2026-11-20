from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.nodes.output_guard import OutputAuditorSchema, output_guard_node
from src.agent.state import LegalAgentState


@pytest.mark.asyncio
@patch("src.agent.nodes.output_guard.get_chat_model")
async def test_output_guard_clear_success(mock_get_chat_model: MagicMock) -> None:
    mock_structured_llm = MagicMock()
    mock_structured_llm.ainvoke = AsyncMock(
        return_value=OutputAuditorSchema(
            analisis_perbandingan="Klaim AI sesuai dokumen.",
            kesimpulan="AMAN",
        )
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_chat_model.return_value = mock_llm

    mock_state: LegalAgentState = {
        "messages": [],
        "search_payloads": [],
        "search_filters": {"is_chitchat": False},
        "final_answer": "Berdasarkan Pasal 156, Anda berhak mendapat pesangon.",
        "retrieved_documents": [
            {
                "pasal": "Pasal 156",
                "text_parent": "Aturan asli pasal 156",
                "text_child": "Hak pesangon",
            }
        ],
        "system_status": "clear",
    }

    node_output = await output_guard_node(mock_state)
    assert node_output["system_status"] == "clear"


@pytest.mark.asyncio
async def test_output_guard_blocked_by_regex() -> None:
    mock_state: LegalAgentState = {
        "messages": [],
        "search_payloads": [],
        "search_filters": {"is_chitchat": False},
        "final_answer": "Berdasarkan Pasal 999 (Fiktif), Anda berhak libur setahun.",
        "retrieved_documents": [
            {
                "pasal": "Pasal 156",
                "text_parent": "Aturan asli pasal 156",
                "text_child": "Hak pesangon",
            }
        ],
        "system_status": "clear",
    }

    node_output = await output_guard_node(mock_state)
    assert node_output["system_status"] == "flagged_output"
    assert "diblokir" in node_output["final_answer"]


@pytest.mark.asyncio
@patch("src.agent.nodes.output_guard.get_chat_model")
async def test_output_guard_blocked_by_llm_semantic(
    mock_get_chat_model: MagicMock,
) -> None:
    mock_structured_llm = MagicMock()
    mock_structured_llm.ainvoke = AsyncMock(
        return_value=OutputAuditorSchema(
            analisis_perbandingan="AI membalikkan makna aturan asli.",
            kesimpulan="HALUSINASI",
        )
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_chat_model.return_value = mock_llm

    mock_state: LegalAgentState = {
        "messages": [],
        "search_payloads": [],
        "search_filters": {"is_chitchat": False},
        "final_answer": (
            "Berdasarkan Pasal 156, Perusahaan bebas memilih untuk tidak "
            "membayar uang pesangon."
        ),
        "retrieved_documents": [
            {
                "pasal": "Pasal 156",
                "text_parent": (
                    "Pengusaha wajib membayar uang pesangon jika terjadi PHK."
                ),
                "text_child": "Kewajiban pembayaran pesangon",
            }
        ],
        "system_status": "clear",
    }

    node_output = await output_guard_node(mock_state)

    assert node_output["system_status"] == "flagged_output"
    assert "tidak lolos uji akurasi" in node_output["final_answer"]
    assert mock_structured_llm.ainvoke.call_count == 1
