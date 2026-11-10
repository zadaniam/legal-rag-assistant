from unittest.mock import AsyncMock, patch

import pytest

from src.agent.nodes.output_guard import output_guard_node
from src.agent.state import LegalAgentState


@pytest.mark.asyncio
@patch("src.agent.nodes.output_guard.ai_client.aio.models.generate_content")
async def test_output_guard_clear_success(mock_generate_content):
    mock_response = AsyncMock()
    mock_response.text = (
        '{"analisis_perbandingan": "Klaim AI sesuai dokumen.", "kesimpulan": "AMAN"}'
    )
    mock_generate_content.return_value = mock_response

    mock_state: LegalAgentState = {
        "search_filters": {"is_chitchat": False},
        "final_answer": "Berdasarkan Pasal 156, Anda berhak mendapat pesangon.",
        "retrieved_documents": [
            {
                "pasal": "Pasal 156",
                "text_parent": "Aturan asli pasal 156",
                "text_child": "Hak pesangon",
            }
        ],
    }

    node_output = await output_guard_node(mock_state)
    assert node_output["system_status"] == "clear"


@pytest.mark.asyncio
async def test_output_guard_blocked_by_regex():
    mock_state: LegalAgentState = {
        "search_filters": {"is_chitchat": False},
        "final_answer": "Berdasarkan Pasal 999 (Fiktif), Anda berhak libur setahun.",
        "retrieved_documents": [
            {
                "pasal": "Pasal 156",
                "text_parent": "Aturan asli pasal 156",
                "text_child": "Hak pesangon",
            }
        ],
    }

    node_output = await output_guard_node(mock_state)
    assert node_output["system_status"] == "flagged_output"
    assert "diblokir" in node_output["final_answer"]


@pytest.mark.asyncio
@patch("src.agent.nodes.output_guard.ai_client.aio.models.generate_content")
async def test_output_guard_blocked_by_llm_semantic(mock_generate_content):
    mock_response = AsyncMock()
    mock_response.text = (
        '{"analisis_perbandingan": "AI membalikkan makna aturan asli.", '
        '"kesimpulan": "HALUSINASI"}'
    )
    mock_generate_content.return_value = mock_response

    mock_state: LegalAgentState = {
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
    }

    node_output = await output_guard_node(mock_state)

    assert node_output["system_status"] == "flagged_output"
    assert "tidak lolos uji akurasi" in node_output["final_answer"]

    assert mock_generate_content.call_count == 1
