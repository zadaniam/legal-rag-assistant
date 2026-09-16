# tests/integration/test_api_integration.py
from unittest.mock import AsyncMock, MagicMock, patch

import httpx  # Impor httpx secara utuh untuk mengambil ASGITransport
import pytest
from httpx import AsyncClient

from src.main import app


@pytest.mark.asyncio
async def test_api_health_check_success():
    """TEST 1: MONITORING HEALTH CHECK ENDPOINT"""
    # PERBAIKAN FORENSIK: Gunakan httpx.ASGITransport untuk membungkus aplikasi FastAPI Anda
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "environment": "development"}


@pytest.mark.asyncio
async def test_api_chat_validation_error_on_empty_payload():
    """TEST 2: VALIDASI SKEMA REQUEST GAGAL (BAD REQUEST / VALIDATION ERROR)"""
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/v1/chat", json={})

    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.asyncio
@patch("src.api.v1.chat.compile_legal_graph")
async def test_api_chat_endpoint_success(mock_compile_legal_graph):
    """TEST 3: SUKSES MENERIMA REQUEST & LOGIKA ROUTER HTTP (HAPPY PATH)"""

    mock_graph = MagicMock()
    mock_ainvoke = AsyncMock()
    mock_ainvoke.return_value = {
        "final_answer": "Halo! Ada yang bisa saya bantu terkait hukum ketenagakerjaan?",
        "system_status": "clear",
        "search_filters": {
            "is_chitchat": True,
            "chitchat_response": "Halo! Ada yang bisa saya bantu terkait hukum ketenagakerjaan?",
        },
        "retrieved_documents": [],
    }
    mock_graph.ainvoke = mock_ainvoke
    mock_compile_legal_graph.return_value = mock_graph

    valid_payload = {
        "message": "Halo AI, selamat siang!",
        "session_id": "sesi_http_test_01",
    }

    # PERBAIKAN FORENSIK: Gunakan httpx.ASGITransport di sini juga
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/v1/chat", json=valid_payload)

    assert response.status_code == 200
    json_data = response.json()
    assert json_data["session_id"] == "sesi_http_test_01"
    assert json_data["system_status"] == "clear"
    assert json_data["is_chitchat"] is True
    assert "citations" in json_data

    assert mock_ainvoke.call_count == 1
