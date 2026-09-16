# src/agent/nodes/context.py

import json

from google import genai
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage
from langsmith.wrappers import wrap_gemini
from pydantic import BaseModel, Field

from src.agent.prompts import CONTEXT_MANAGER_SYSTEM_PROMPT
from src.agent.state import LegalAgentState
from src.config import settings
from src.logger import logger


# 1. Definisikan Skema Data Menggunakan Pydantic (Standar Struktur Output)
class SearchQueryPayload(BaseModel):
    query: str = Field(
        description="Kata kunci pencarian spesifik hukum, disarikan dari pertanyaan user."
    )
    target_uu: str = Field(
        description="Nama undang-undang yang ditargetkan, pilih hanya 'UU Cipta Kerja', 'UU Ketenagakerjaan', atau 'Semua' jika berpotensi ada di keduanya."
    )


class RouterOutputSchema(BaseModel):
    is_chitchat: bool = Field(
        description="True jika user hanya menyapa, berterima kasih, atau basa-basi sosial tanpa konteks hukum."
    )
    chitchat_response: str | None = Field(
        None,
        description="Respons sapaan langsung yang ramah dan profesional jika is_chitchat bernilai True.",
    )
    search_payloads: list[SearchQueryPayload] = Field(
        default=[],
        description="Daftar objek query pencarian paralel jika is_chitchat bernilai False.",
    )


# 2. Inisialisasi Google Gen AI Client Resmi (Menggunakan Kunci Rahasia .env)
ai_client = wrap_gemini(
    genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())
)


async def context_manager_node(state: LegalAgentState) -> dict:
    logger.info("=== Node 2: Mengeksekusi Context Manager (Gemini Router) ===")

    # 1. Ambil seluruh riwayat pesan dari state
    all_messages = state.get("messages", [])

    # 2. Ambil 5 turn terakhir (maksimal 10 pesan: 5 User + 5 AI) sebelum pesan terakhir saat ini
    # Kita potong riwayatnya, tapi kecualikan pesan paling akhir (karena itu input user saat ini)
    recent_history_messages = all_messages[:-1][-10:] if len(all_messages) > 1 else []

    # 3. Format riwayat pesan pendek tersebut menjadi string teks terstruktur
    formatted_recent_history = ""
    for msg in recent_history_messages:
        if isinstance(msg, HumanMessage):
            formatted_recent_history += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            formatted_recent_history += f"Asisten: {msg.content}\n"

    # 4. Ambil pesan terakhir dari user (input aktif saat ini)
    user_message = all_messages[-1].content

    # 5. Susun instruksi lengkap dengan menyuapkan 5 turn chat terakhir sebagai konteks berjalan
    full_system_instruction = CONTEXT_MANAGER_SYSTEM_PROMPT
    if formatted_recent_history:
        full_system_instruction += f"\n\n[Konteks Tambahan] 5 Turn Percakapan Terakhir:\n{formatted_recent_history}"

    try:
        # Panggil Gemini dengan konfigurasi JSON Schema yang ketat
        response = await ai_client.aio.models.generate_content(
            model="gemini-3.5-flash-lite",  # Cepat, murah, dan sangat cerdas untuk tugas klasifikasi/routing
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=full_system_instruction,
                temperature=0.0,  # Deterministic (0.0 agar jawaban tidak berubah-ubah)
                response_mime_type="application/json",
                response_schema=RouterOutputSchema,
            ),
        )

        # Ambil hasil JSON mentah dari Gemini dan ubah menjadi dictionary Python
        result_json = json.loads(response.text)
        logger.info(
            f"Gemini Router Berhasil Memproses Rute. is_chitchat={result_json.get('is_chitchat')}"
        )

        # Masukkan hasil analisis ke dalam state pencarian graf
        return {
            "search_payloads": result_json.get("search_payloads", []),
            "search_filters": result_json,
        }

    except Exception as e:
        logger.error(f"Gagal memproses routing semantik di Node 2: {e}")
        # Mekanisme Fallback Darurat: Anggap sebagai pencarian umum jika AI gagal merespons format JSON
        fallback_payload = [{"query": user_message, "target_uu": "Semua"}]
        return {
            "search_payloads": fallback_payload,
            "search_filters": {"is_chitchat": False, "chitchat_response": None},
        }
