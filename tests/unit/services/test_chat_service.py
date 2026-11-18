"""Unit tests for ChatService SSE streaming and payload formatting."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.schemas.chat import ChatRequest
from src.services.chat_service import ChatService


def _parse_sse_event(raw_sse: str) -> dict:
    cleaned = raw_sse.strip().removeprefix("data: ")
    return json.loads(cleaned)


@pytest.mark.asyncio
async def test_stream_chat_response_success_with_citations(mocker) -> None:

    async def mock_astream(*args, **kwargs):
        yield {}  # Empty chunk should be safely skipped
        yield {"input_guard": {"system_status": "clear"}}
        yield {"legal_search": {"retrieved_documents": []}}
        yield {"output_guard": {"final_answer": "Berdasarkan Pasal 14 UU BPJS..."}}

    mock_snapshot = MagicMock()
    mock_snapshot.values = {
        "final_answer": "Berdasarkan Pasal 14 UU BPJS...",
        "system_status": "clear",
        "search_filters": {"is_chitchat": False},
        "retrieved_documents": [
            {
                "uu_id": "uu_bpjs_2011",
                "bab": "BAB IV",
                "pasal": 14,
                "text_child": "Setiap orang wajib menjadi Peserta...",
                "text_parent": "Pasal 14 Setiap orang termasuk orang asing...",
            }
        ],
    }

    mock_graph = MagicMock()
    mock_graph.astream = mock_astream
    mock_graph.aget_state = AsyncMock(return_value=mock_snapshot)

    mocker.patch(
        "src.services.chat_service.compile_legal_graph", return_value=mock_graph
    )

    payload = ChatRequest(
        message="Siapa saja yang wajib ikut BPJS?",
        session_id="unit_session_01",
    )

    events = [
        _parse_sse_event(raw_chunk)
        async for raw_chunk in ChatService.stream_chat_response(payload)
    ]

    assert len(events) == 4
    assert events[0] == {"event": "node_complete", "node": "input_guard"}
    assert events[1] == {"event": "node_complete", "node": "legal_search"}
    assert events[2] == {"event": "node_complete", "node": "output_guard"}

    final_event = events[3]
    assert final_event["event"] == "final_result"
    assert final_event["session_id"] == "unit_session_01"
    assert final_event["final_answer"] == "Berdasarkan Pasal 14 UU BPJS..."
    assert final_event["system_status"] == "clear"
    assert final_event["is_chitchat"] is False
    assert len(final_event["citations"]) == 1
    assert final_event["citations"][0]["uu_id"] == "uu_bpjs_2011"
    assert final_event["citations"][0]["pasal"] == "14"


@pytest.mark.asyncio
async def test_stream_chat_response_emits_error_event_on_exception(mocker) -> None:

    async def failing_astream(*args, **kwargs):
        yield {"input_guard": {"system_status": "clear"}}
        raise RuntimeError("Qdrant connection refused")

    mock_graph = MagicMock()
    mock_graph.astream = failing_astream

    mocker.patch(
        "src.services.chat_service.compile_legal_graph", return_value=mock_graph
    )

    payload = ChatRequest(
        message="Apa sanksi tidak membayar iuran?",
        session_id="unit_session_error",
    )

    events = [
        _parse_sse_event(raw_chunk)
        async for raw_chunk in ChatService.stream_chat_response(payload)
    ]

    assert len(events) == 2
    assert events[0] == {"event": "node_complete", "node": "input_guard"}
    assert events[1]["event"] == "error"
    assert "Qdrant connection refused" in events[1]["detail"]
