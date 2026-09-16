# src/agent/nodes/reasoner.py

from google import genai
from google.genai import types
from langsmith.wrappers import wrap_gemini

from src.agent.prompts import LEGAL_REASONER_SYSTEM_PROMPT
from src.agent.state import LegalAgentState
from src.config import settings
from src.logger import logger

# Inisialisasi Google Gen AI Client Resmi
ai_client = wrap_gemini(
    genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())
)


async def legal_reasoner_node(state: LegalAgentState) -> dict:
    logger.info("=== Node 4: Mengeksekusi Legal Reasoner (Gemini Pro) ===")

    # 1. Cek Rute Khusus: Jika state mendeteksi ini adalah Chit-Chat, langsung ambil respon dari router
    filters = state.get("search_filters", {})
    if filters.get("is_chitchat") == True:
        logger.info("Meneruskan respon Chit-Chat langsung ke pengguna.")
        chitchat_ans = filters.get(
            "chitchat_response", "Halo! Ada yang bisa saya bantu terkait hukum?"
        )
        return {"final_answer": chitchat_ans, "retrieved_documents": []}

    # 2. Ambil data dokumen hukum hasil retrieval Node 3
    retrieved_docs = state.get("retrieved_documents", []) or []
    user_message = state["messages"][-1].content

    # Jika karena suatu hal tidak ada dokumen hukum yang terambil
    if not retrieved_docs:
        logger.warning(
            "Node Reasoner mendeteksi tidak ada dokumen hukum yang disediakan."
        )
        return {
            "final_answer": "Maaf, saya tidak menemukan dokumen rujukan yang relevan di basis data saya untuk menjawab pertanyaan Anda."
        }

    # Format kumpulan pasal menjadi teks terstruktur untuk disuapkan ke System Prompt
    formatted_docs = ""
    for idx, doc in enumerate(retrieved_docs, start=1):
        formatted_docs += f"\nDokumen #{idx}:\nRegulasi: {doc.get('nama_regulasi', 'UU')}\nPasal: {doc.get('pasal')}\nIsi Teks:\n{doc.get('text_parent')}\n"

    # Susun instruksi emas penolak halusinasi
    full_system_instruction = LEGAL_REASONER_SYSTEM_PROMPT.format(
        retrieved_documents=formatted_docs
    )

    try:
        # Panggil Gemini Pro (Kasta tertinggi untuk penalaran logika hukum terstruktur)
        response = await ai_client.aio.models.generate_content(
            model="gemini-3.5-flash-lite",  # Sangat patuh pada instruksi prompt yang panjang dan restriktif
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=full_system_instruction,
                temperature=0.2,  # Rendah agar kreatifitas ditekan dan fokus pada teks pasal asli
            ),
        )

        logger.info("Gemini Pro berhasil merumuskan argumen analisis hukum.")
        return {"final_answer": response.text.strip()}

    except Exception as e:
        logger.error(f"Gagal memproses penyusunan argumen hukum di Node 4: {e}")
        return {
            "final_answer": "Maaf, terjadi kendala teknis internal saat menganalisis argumen hukum. Silakan coba beberapa saat lagi."
        }
