"""Output guard node for the legal RAG agent.

Validates AI-generated legal answers for hallucination using a hybrid
approach: deterministic regex pasal citation check followed by semantic
evaluation with Gemini.
"""

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
    """Schema for Gemini output auditor evaluation result."""

    analisis_perbandingan: str = Field(
        description="Objective claim-by-claim evaluation analysis."
    )
    kesimpulan: str = Field(description="Only contains string 'AMAN' or 'HALUSINASI'")


ai_client = wrap_gemini(
    genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())
)


async def output_guard_node(state: LegalAgentState) -> dict:
    """Validate AI answer for hallucination using regex and Gemini."""
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

        result_json = json.loads(response.text)
        evaluation_result = result_json.get("kesimpulan", "AMAN").strip().upper()

        logger.info(f"Auditor analysis: {result_json.get('analisis_perbandingan')}")
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
