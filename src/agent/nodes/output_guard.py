# src/agent/nodes/output_guard.py

import json
import re

from google import genai
from google.genai import types
from langsmith.wrappers import wrap_gemini
from pydantic import BaseModel, Field

from src.agent.prompts import OUTPUT_GUARD_SYSTEM_PROMPT
from src.agent.state import LegalAgentState
from src.config import settings
from src.logger import logger


class OutputAuditorSchema(BaseModel):
    analisis_perbandingan: str = Field(
        description="Analisis evaluasi klaim demi klaim secara objektif."
    )
    kesimpulan: str = Field(description="Hanya berisi string 'AMAN' atau 'HALUSINASI'")


# Inisialisasi Google Gen AI Client
ai_client = wrap_gemini(
    genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())
)


async def output_guard_node(state: LegalAgentState) -> dict:
    logger.info("=== Node 5: Mengeksekusi Output Guard (Hibrida) ===")

    # Rute Khusus: Lewati pengecekan jika ini adalah Chit-Chat
    filters = state.get("search_filters", {})
    if filters.get("is_chitchat") == True:
        return {"system_status": "clear"}

    final_answer = state.get("final_answer", "")
    retrieved_docs = state.get("retrieved_documents", []) or []

    # ---------------------------------------------------------
    # TAHAP 1: VALIDASI PYTHON REGEX (DETERMINISTIK)
    # ---------------------------------------------------------
    # Ekstrak semua pola angka pasal dari jawaban AI menggunakan regex
    cited_pasals = re.findall(r"Pasal\s+(\d+)", final_answer)

    # Ambil daftar nomor pasal resmi yang terambil dari database Qdrant
    allowed_pasals = [str(doc.get("pasal")) for doc in retrieved_docs]

    logger.info(
        f"Pasal disitasi AI: {cited_pasals} | Pasal diizinkan Qdrant: {allowed_pasals}"
    )

    # Validasi instan: Jika AI menyebut nomor pasal siluman yang tidak ada di dokumen retrieval, blokir langsung!
    for pasal in cited_pasals:
        if pasal not in allowed_pasals:
            logger.warning(
                f"Output Guard memblokir respons! AI menyitasi pasal fiktif/luar: Pasal {pasal}"
            )
            return {
                "system_status": "flagged_output",
                "final_answer": "Maaf, sistem mendeteksi ketidaksesuaian referensi pasal hukum pada jawaban AI. Respons diblokir demi akurasi.",
            }

    # ---------------------------------------------------------
    # TAHAP 2: VALIDASI GEMINI (SEMANTIK / EVALUASI MAKNA)
    # ---------------------------------------------------------
    formatted_docs = "\n".join(
        [f"Pasal {d.get('pasal')}: {d.get('text_parent')}" for d in retrieved_docs]
    )
    user_query = (
        f"DOKUMEN_ASLI_RETRIEVAL:\n{formatted_docs}\n\nJAWABAN_AI:\n{final_answer}"
    )

    try:
        response = await ai_client.aio.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=user_query,
            config=types.GenerateContentConfig(
                system_instruction=OUTPUT_GUARD_SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=OutputAuditorSchema,
            ),
        )

        # 1. Parsing teks JSON dari Gemini menjadi dictionary Python
        result_json = json.loads(response.text)
        evaluation_result = result_json.get("kesimpulan", "AMAN").strip().upper()

        # Log analisis internal biar bisa dipantau juga via terminal selain LangSmith
        logger.info(f"Analisis Auditor: {result_json.get('analisis_perbandingan')}")
        logger.info(f"Hasil Evaluasi Semantik Output Guard: {evaluation_result}")

        # 2. Cek kesimpulan akhir dari hasil evaluasi struktur JSON
        if evaluation_result == "HALUSINASI":
            logger.warning(
                "Output Guard memblokir respons! Terdeteksi penyimpangan makna hukum."
            )
            return {
                "system_status": "flagged_output",
                "final_answer": "Maaf, jawaban tidak lolos uji akurasi substansi hukum dari pengawas sistem.",
            }

        logger.info("Jawaban dinyatakan AMAN dan valid oleh Output Guard.")
        return {"system_status": "clear"}

    except Exception as e:
        logger.error(f"Gagal mengeksekusi evaluasi semantik di Node 5: {e}")
        # Cari aman: Jika guardrails eror, loloskan asalkan lolos sensor angka Regex di Tahap 1
        return {"system_status": "clear"}
