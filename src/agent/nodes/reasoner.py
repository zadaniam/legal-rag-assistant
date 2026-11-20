"""Legal reasoner node for the legal RAG agent.

Generates legal answers with citations using a provider-agnostic chat model,
grounded in retrieved legal documents from Qdrant.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.agent.configuration import AgentConfiguration
from src.agent.prompts import LEGAL_REASONER_SYSTEM_PROMPT
from src.agent.state import LegalAgentState
from src.database.vector_schema import PayloadField
from src.logger import logger
from src.utils.llm import get_chat_model


async def legal_reasoner_node(
    state: LegalAgentState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Generate legal answer with citations using the configured chat model."""
    filters = state.get("search_filters", {})
    if filters.get("is_chitchat"):
        logger.info("Forwarding chitchat response directly to user.")
        chitchat_ans = filters.get(
            "chitchat_response", "Halo! Ada yang bisa saya bantu terkait hukum?"
        )
        return {"final_answer": chitchat_ans, "retrieved_documents": []}

    retrieved_docs = state.get("retrieved_documents", []) or []
    user_message = str(state["messages"][-1].content)

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
        llm = get_chat_model(
            model=agent_cfg.reasoner_model,
            temperature=agent_cfg.reasoner_temperature,
        )
        messages = [
            SystemMessage(content=full_system_instruction),
            HumanMessage(content=user_message),
        ]
        response = await llm.ainvoke(messages)

        logger.info("Chat model successfully formulated legal analysis argument.")
        return {"final_answer": str(response.content).strip()}

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
