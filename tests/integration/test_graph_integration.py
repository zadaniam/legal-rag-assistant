# tests/integration/test_graph_integration.py
import re

import pytest
from langchain_core.messages import HumanMessage

from src.agent.graph import (
    compile_legal_graph,  # PERBAIKAN 1: Impor fungsi kompilasi graf
)
from src.logger import logger

# Pastikan unit test tunduk pada cakupan event loop tunggal sesi agar aman
pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_legal_assistant_e2e_substance_query():
    logger.info("=== MEMULAI ASYNC INTEGRATION TEST AKHIR (END-TO-END) ===")

    # 1. Kompilasi graf secara aman di dalam event loop pengujian yang aktif
    legal_rag_graph = compile_legal_graph()

    # 2. Siapkan input pertanyaan hukum nyata yang datanya sudah di-ingest ke Qdrant sebelumnya
    input_state = {
        "messages": [
            HumanMessage(
                content="Berapa lama uang pesangon yang saya dapatkan kalau baru bekerja kurang dari 1 tahun menurut undang-undang ketenagakerjaan?"
            )
        ]
    }

    # Konfigurasikan thread ID sesi baru agar terisolasi di AsyncPostgres Checkpointer
    config = {"configurable": {"thread_id": "sesi_uji_integrasi_e2e_async_001"}}

    # 3. PERBAIKAN 3: Jalankan Graf secara utuh menggunakan 'await' dan '.ainvoke'
    result = await legal_rag_graph.ainvoke(input_state, config=config)

    # 4. VALIDASI INTEGRAL (ASSERTS)
    # A. Pastikan sistem status berakhir dengan "clear" (lolos Input Guard & Output Guard)
    assert result["system_status"] == "clear"

    # B. Pastikan Node 2 sukses mengekstrak query dan mendeteksi bukan chitchat
    search_filters = result.get("search_filters", {})
    assert search_filters.get("is_chitchat") is False
    assert len(result["search_payloads"]) >= 1

    # C. Pastikan Node 3 sukses menembus Qdrant Async dan mengambil dokumen pasal asli
    assert len(result["retrieved_documents"]) >= 1

    # 🎯 PERBAIKAN ASERSI: Menggunakan kunci 'pasal' dari payload dictionary hasil modifikasi DB terbaru
    first_doc = result["retrieved_documents"][0]
    assert "pasal" in first_doc
    assert "uu_id" in first_doc

    # D. Pastikan Node 4 (Gemini Pro) menjawab dengan tepat berdasarkan data ingest ("1 bulan upah")
    final_answer = result["final_answer"]
    logger.info(f"JAWABAN INTEGRASI AKHIR AI:\n{final_answer}")

    # Mengecek fleksibilitas tulisan "1 bulan" atau "satu bulan" di dalam jawaban LLM
    match_result = re.search(r"(1|satu).*?bulan", final_answer.lower())
    assert match_result, (
        f"Gagal memvalidasi jawaban! AI harusnya menyebutkan durasi '1 bulan' atau 'satu bulan'. "
        f"Jawaban asli AI: '{final_answer}'"
    )

    # Pastikan AI menyertakan sitasi wajib tanda kurung kotak sesuai instruksi prompt
    assert "[UU" in final_answer or "Pasal" in final_answer
