# tests/integration/test_graph_integration.py
import re

import pytest
from langchain_core.messages import HumanMessage

from src.agent.graph import (
    compile_legal_graph,  # PERBAIKAN 1: Impor fungsi kompilasi graf
)
from src.logger import logger


# PERBAIKAN 2: Kembalikan dekorator asyncio karena graf kita sekarang asinkronus penuh
@pytest.mark.asyncio
async def test_legal_assistant_e2e_substance_query():
    logger.info("=== MEMULAI ASYNC INTEGRATION TEST AKHIR (END-TO-END) ===")

    # 1. Kompilasi graf secara aman di dalam event loop pengujian yang aktif
    legal_rag_graph = compile_legal_graph()

    # 2. Siapkan input pertanyaan hukum nyata yang datanya sudah di-ingest ke Qdrant sebelumnya
    input_state = {
        "messages": [
            HumanMessage(
                content="Berapa lama uang pesangon yang saya dapatkan kalau baru bekerja kurang dari 1 tahun menurut undang-undang cipta kerja?"
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
    assert str(result["retrieved_documents"][0]["pasal"]) == "40"

    # D. Pastikan Node 4 (Gemini Pro) menjawab dengan tepat berdasarkan data ingest ("1 bulan upah")
    final_answer = result["final_answer"]
    logger.info(f"JAWABAN INTEGRASI AKHIR AI:\n{final_answer}")

    assert re.search(r"1.*bulan", final_answer.lower()) is not None
    # Pastikan AI menyertakan sitasi wajib tanda kurung kotak sesuai instruksi prompt
    assert "[UU" in final_answer or "Pasal" in final_answer
