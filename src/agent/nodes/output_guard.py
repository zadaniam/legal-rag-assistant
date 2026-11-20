"""Output guard node for the legal RAG agent.

Validates AI-generated legal answers for hallucination using a hybrid
approach: deterministic regex pasal citation check followed by semantic
evaluation with a structured LLM auditor.
"""

import re
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.agent.configuration import AgentConfiguration
from src.agent.prompts import OUTPUT_GUARD_SYSTEM_PROMPT
from src.agent.state import LegalAgentState
from src.logger import logger
from src.utils.llm import get_chat_model


class OutputAuditorSchema(BaseModel):
    """Schema for LLM output auditor evaluation result."""

    analisis_perbandingan: str = Field(
        description="Objective claim-by-claim evaluation analysis."
    )
    kesimpulan: str = Field(description="Only contains string 'AMAN' or 'HALUSINASI'")


async def output_guard_node(
    state: LegalAgentState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Validate AI answer for hallucination using regex and structured LLM audit."""
    filters = state.get("search_filters", {})
    if filters.get("is_chitchat"):
        return {"system_status": "clear"}

    final_answer = state.get("final_answer", "")
    retrieved_docs = state.get("retrieved_documents", []) or []

    cited_pasals = re.findall(r"Pasal\s+(\d+)", final_answer)

    allowed_pasals = []
    for doc in retrieved_docs:
        pasal_text = str(doc.get("pasal", ""))
        match = re.search(r"\d+", pasal_text)
        if match:
            allowed_pasals.append(match.group())

    logger.info(
        "Pasal cited by AI: "
        f"{cited_pasals} | Pasal allowed from Qdrant: {allowed_pasals}"
    )

    # Block immediately if AI cites a phantom pasal not present in retrieved documents.
    for pasal in cited_pasals:
        if pasal not in allowed_pasals:
            logger.warning(
                "Output Guard blocking response! AI cited "
                f"fictitious/out-of-scope pasal: Pasal {pasal}"
            )
            return {
                "system_status": "flagged_output",
                "final_answer": (
                    "Maaf, sistem mendeteksi ketidaksesuaian referensi pasal "
                    "hukum pada jawaban AI. Respons diblokir demi akurasi."
                ),
            }

    agent_cfg = AgentConfiguration.from_runnable_config(config)

    formatted_docs = "\n".join(
        [
            f"{d.get('pasal', 'Pasal N/A')}: "
            f"{d.get('text_parent', d.get('text_child', ''))}"
            for d in retrieved_docs
        ]
    )
    user_query = (
        f"DOKUMEN_ASLI_RETRIEVAL:\n{formatted_docs}\n\nJAWABAN_AI:\n{final_answer}"
    )

    try:
        llm = get_chat_model(
            model=agent_cfg.auditor_model,
            temperature=agent_cfg.auditor_temperature,
        )
        structured_llm = llm.with_structured_output(OutputAuditorSchema)

        messages = [
            SystemMessage(content=OUTPUT_GUARD_SYSTEM_PROMPT),
            HumanMessage(content=user_query),
        ]
        result = cast(OutputAuditorSchema, await structured_llm.ainvoke(messages))
        evaluation_result = result.kesimpulan.strip().upper()

        logger.info(f"Auditor analysis: {result.analisis_perbandingan}")
        logger.info(f"Output Guard semantic evaluation result: {evaluation_result}")

        if evaluation_result == "HALUSINASI":
            logger.warning(
                "Output Guard blocking response! Detected deviation in legal substance."
            )
            return {
                "system_status": "flagged_output",
                "final_answer": (
                    "Maaf, jawaban tidak lolos uji akurasi substansi hukum "
                    "dari pengawas sistem."
                ),
            }

        logger.info("Answer declared safe and valid by Output Guard.")
        return {"system_status": "clear"}

    except Exception as e:
        logger.error(f"Failed to execute semantic evaluation in output guard node: {e}")
        # Fail safe: allow through if regex pasal check in stage 1 passed.
        return {"system_status": "clear"}
