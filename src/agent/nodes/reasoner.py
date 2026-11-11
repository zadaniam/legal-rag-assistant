"""Legal reasoner node for the legal RAG agent.

Generates legal answers with citations using Gemini, grounded in retrieved
legal documents from Qdrant.
"""

from google import genai
from google.genai import types
from langsmith.wrappers import wrap_gemini

from src.agent.prompts import LEGAL_REASONER_SYSTEM_PROMPT
from src.agent.state import LegalAgentState
from src.config import settings
from src.logger import logger

ai_client = wrap_gemini(
    genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())
)


async def legal_reasoner_node(state: LegalAgentState) -> dict:
    """Generate legal answer with citations using Gemini."""
    filters = state.get("search_filters", {})
    if filters.get("is_chitchat"):
        logger.info("Forwarding chitchat response directly to user.")
        chitchat_ans = filters.get(
            "chitchat_response", "Halo! Ada yang bisa saya bantu terkait hukum?"
        )
        return {"final_answer": chitchat_ans, "retrieved_documents": []}

    retrieved_docs = state.get("retrieved_documents", []) or []
    user_message = state["messages"][-1].content

    if not retrieved_docs:
        logger.warning("Reasoner node detected no legal documents provided.")
        return {
            "final_answer": (
                "Maaf, saya tidak menemukan dokumen rujukan yang relevan di "
                "basis data saya untuk menjawab pertanyaan Anda."
            )
        }

    formatted_docs = ""
    for idx, doc in enumerate(retrieved_docs, start=1):
        formatted_docs += (
            f"\nDokumen #{idx}:\n"
            f"ID UU: {doc.get('uu_id', 'N/A')}\n"
            f"Bab: {doc.get('bab', 'N/A')}\n"
            f"Pasal: {doc.get('pasal', 'N/A')}\n"
            f"Isi Konteks Hukum:\n{doc.get('text_parent', doc.get('text_child', ''))}\n"
        )

    full_system_instruction = LEGAL_REASONER_SYSTEM_PROMPT.format(
        retrieved_documents=formatted_docs
    )

    try:
        response = await ai_client.aio.models.generate_content(
            model="gemini-3.5-flash-lite",  # Compliant with restrictive instructions
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=full_system_instruction,
                temperature=0.2,  # Low temperature: focus on original legal text
            ),
        )

        logger.info("Gemini Pro successfully formulated legal analysis argument.")
        return {"final_answer": response.text.strip()}

    except Exception as e:
        logger.error(
            f"Failed to process legal argument synthesis in reasoner node: {e}"
        )
        return {
            "final_answer": (
                "Maaf, terjadi kendala teknis internal saat menganalisis "
                "argumen hukum. Silakan coba beberapa saat lagi."
            )
        }
