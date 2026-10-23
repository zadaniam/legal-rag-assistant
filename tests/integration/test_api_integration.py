# tests/integration/test_api_integration.py
from unittest.mock import MagicMock, patch

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


# tests/integration/test_api_integration.py
import json

import pytest

# Asumsikan 'app' diimpor dari modul utama aplikasi Anda


@pytest.mark.asyncio
@patch("src.api.v1.chat.compile_legal_graph")
async def test_api_chat_endpoint_success(mock_compile_legal_graph):
    """TEST 3: SUKSES MENERIMA REQUEST & LOGIKA ROUTER HTTP (HAPPY PATH VIA SSE STREAM)"""

    # 1. Mock LangGraph .astream menggunakan async generator sebaris
    async def mock_astream(*args, **kwargs):
        yield {
            "output_guard": {
                "final_answer": "Halo! Ada yang bisa saya bantu terkait hukum ketenagakerjaan?",
                "system_status": "clear",
                "search_filters": {"is_chitchat": True},
                "retrieved_documents": [],
            }
        }

    mock_graph = MagicMock()
    mock_graph.astream = mock_astream
    mock_compile_legal_graph.return_value = mock_graph

    # 2. Tembak HTTP API via httpx ASGITransport
    async with AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post(
            "/api/v1/chat",
            json={
                "message": "Halo AI, selamat siang!",
                "session_id": "sesi_http_test_01",
            },
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # 3. Ekstrak data JSON final_result secara instan dengan regex/substring sederhana
    final_result_line = [
        line for line in response.text.splitlines() if 'event": "final_result"' in line
    ][0]
    json_data = json.loads(final_result_line.replace("data: ", "", 1))

    # 4. Validasi Integral
    assert json_data["session_id"] == "sesi_http_test_01"
    assert json_data["system_status"] == "clear"
    assert json_data["is_chitchat"] is True
    assert "citations" in json_data
