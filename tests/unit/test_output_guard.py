# tests/unit/test_output_guard.py
from unittest.mock import AsyncMock, patch

import pytest

from src.agent.nodes.output_guard import output_guard_node
from src.agent.state import LegalAgentState


# TEST 1: Kasus AI Lolos Sensor (Happy Path)
@pytest.mark.asyncio
# PERBAIKAN 1: Pindahkan jalur target patch ke submodule '.aio.'
@patch("src.agent.nodes.output_guard.ai_client.aio.models.generate_content")
async def test_output_guard_clear_success(mock_generate_content):
    # PERBAIKAN 2: Gunakan AsyncMock agar objek respons bisa di-await
    mock_response = AsyncMock()
    mock_response.text = "AMAN"
    mock_generate_content.return_value = mock_response

    mock_state: LegalAgentState = {
        "search_filters": {"is_chitchat": False},
        "final_answer": "Berdasarkan Pasal 156, Anda berhak mendapat pesangon.",
        "retrieved_documents": [
            {"pasal": "156", "text_parent": "Aturan asli pasal 156"}
        ],
    }

    # PERBAIKAN 3: Tambahkan 'await' saat mengeksekusi node
    node_output = await output_guard_node(mock_state)
    assert node_output["system_status"] == "clear"


# TEST 2: Kasus AI Diblokir oleh Sensor Angka Python Regex (Pasal Siluman)
@pytest.mark.asyncio  # PERBAIKAN 4: Tambahkan decorator async karena fungsinya sekarang coroutine
async def test_output_guard_blocked_by_regex():
    mock_state: LegalAgentState = {
        "search_filters": {"is_chitchat": False},
        "final_answer": "Berdasarkan Pasal 999 (Fiktif), Anda berhak libur setahun.",
        "retrieved_documents": [
            {"pasal": "156", "text_parent": "Aturan asli pasal 156"}
        ],
    }

    # PERBAIKAN 5: Tambahkan 'await' saat mengeksekusi node
    node_output = await output_guard_node(mock_state)
    assert node_output["system_status"] == "flagged_output"
    assert "diblokir" in node_output["final_answer"]


# TEST 3: Kasus AI Diblokir oleh Evaluasi Semantik LLM (Nomor Pasal Benar, tapi Makna Hukum Ngawur)
@pytest.mark.asyncio
# PERBAIKAN 6: Pindahkan jalur target patch ke submodule '.aio.'
@patch("src.agent.nodes.output_guard.ai_client.aio.models.generate_content")
async def test_output_guard_blocked_by_llm_semantic(mock_generate_content):
    # 1. Simulasikan Gemini Router mendeteksi adanya penyimpangan makna hukum secara async
    mock_response = AsyncMock()
    mock_response.text = "HALUSINASI"
    mock_generate_content.return_value = mock_response

    # 2. Siapkan state di mana nomor pasal BENAR (Pasal 156 ada di retrieval),
    # tetapi isi jawabannya memanipulasi hukum (menyebut perusahaan bisa mangkir bayar)
    mock_state: LegalAgentState = {
        "search_filters": {"is_chitchat": False},
        "final_answer": "Berdasarkan Pasal 156, Perusahaan bebas memilih untuk tidak membayar uang pesangon.",
        "retrieved_documents": [
            {
                "pasal": "156",
                "text_parent": "Pengusaha wajib membayar uang pesangon jika terjadi PHK.",
            }
        ],
    }

    # 3. PERBAIKAN 7: Jalankan fungsi Node menggunakan kata kunci 'await'
    node_output = await output_guard_node(mock_state)

    # 4. Validasi: Sistem harus mendeteksi bendera bahaya (flagged_output) karena sensor semantik LLM
    assert node_output["system_status"] == "flagged_output"
    assert "tidak lolos uji akurasi" in node_output["final_answer"]

    # Pastikan API Gemini benar-benar terpanggil karena sensor Regex di awal berhasil lolos
    assert mock_generate_content.call_count == 1
