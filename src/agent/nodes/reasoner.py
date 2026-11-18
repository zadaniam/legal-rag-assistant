"""Legal reasoner node for the legal RAG agent.

Generates legal answers with citations using Gemini, grounded in retrieved
legal documents from Qdrant.
"""

from google import genai
from google.genai import types
from langchain_core.runnables import RunnableConfig
from langsmith.wrappers import wrap_gemini

from src.agent.configuration import AgentConfiguration
from src.agent.prompts import LEGAL_REASONER_SYSTEM_PROMPT
from src.agent.state import LegalAgentState
from src.config import settings
from src.database.vector_schema import PayloadField
from src.logger import logger

ai_client = wrap_gemini(
    genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())
)


async def legal_reasoner_node(
    state: LegalAgentState, config: RunnableConfig | None = None
) -> dict:
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

    agent_cfg = AgentConfiguration.from_runnable_config(config)

    formatted_docs = ""
    for idx, doc in enumerate(retrieved_docs, start=1):
        content_text = doc.get(
            PayloadField.TEXT_PARENT.value,
            doc.get(PayloadField.TEXT_CHILD.value, ""),
        )
        formatted_docs += (
            f"\nDokumen #{idx}:\n"
            f"ID UU: {doc.get(PayloadField.UU_ID.value, 'N/A')}\n"
            f"Bab: {doc.get(PayloadField.BAB.value, 'N/A')}\n"
            f"Pasal: {doc.get(PayloadField.PASAL.value, 'N/A')}\n"
            f"Isi Konteks Hukum:\n{content_text}\n"
        )

    full_system_instruction = LEGAL_REASONER_SYSTEM_PROMPT.format(
        retrieved_documents=formatted_docs
    )

    try:
        response = await ai_client.aio.models.generate_content(
            model=agent_cfg.reasoner_model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=full_system_instruction,
                temperature=agent_cfg.reasoner_temperature,
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
