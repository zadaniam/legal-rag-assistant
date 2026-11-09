"""Context manager node for the legal RAG agent.

Routes user input to either chitchat response or legal search based on
semantic classification using Gemini.
"""

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


class SearchQueryPayload(BaseModel):
    """Schema for a single search query payload extracted from user question."""

    query: str = Field(
        description="Specific legal search keyword extracted from the user question."
    )
    target_uu: str = Field(
        description=(
            "Targeted law ID. Choose only 'uu_ketenagakerjaan_2003', "
            "'uu_bpjs_2011', or 'Semua' if potentially in both."
        )
    )


class RouterOutputSchema(BaseModel):
    """Schema for Gemini router output classifying user input."""

    is_chitchat: bool = Field(
        description=(
            "True if user only greets, thanks, or makes small talk "
            "without legal context."
        )
    )
    chitchat_response: str | None = Field(
        None,
        description=(
            "Direct friendly and professional greeting response if is_chitchat is True."
        ),
    )
    search_payloads: list[SearchQueryPayload] = Field(
        default=[],
        description="List of parallel search query objects if is_chitchat is False.",
    )


ai_client = wrap_gemini(
    genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())
)


async def context_manager_node(state: LegalAgentState) -> dict:
    """Classify user input and extract search payloads using Gemini router."""
    all_messages = state.get("messages", [])

    # Keep only the last 10 messages (5 user + 5 AI) before the current input,
    # excluding the most recent message which is the current user input.
    recent_history_messages = all_messages[:-1][-10:] if len(all_messages) > 1 else []

    formatted_recent_history = ""
    for msg in recent_history_messages:
        if isinstance(msg, HumanMessage):
            formatted_recent_history += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            formatted_recent_history += f"Asisten: {msg.content}\n"

    user_message = all_messages[-1].content

    full_system_instruction = CONTEXT_MANAGER_SYSTEM_PROMPT
    if formatted_recent_history:
        full_system_instruction += (
            f"\n\n[Konteks Tambahan] 5 Turn Percakapan Terakhir:\n"
            f"{formatted_recent_history}"
        )

    try:
        response = await ai_client.aio.models.generate_content(
            model="gemini-3.5-flash-lite",  # Fast and smart for classification/routing
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=full_system_instruction,
                temperature=0.0,  # Deterministic output: 0.0 prevents varying answers
                response_mime_type="application/json",
                response_schema=RouterOutputSchema,
            ),
        )

        result_json = json.loads(response.text)
        logger.info(
            "Gemini Router processed route successfully. "
            f"is_chitchat={result_json.get('is_chitchat')}"
        )

        return {
            "search_payloads": result_json.get("search_payloads", []),
            "search_filters": result_json,
        }

    except Exception as e:
        logger.error(f"Failed to process semantic routing in context manager node: {e}")
        # Emergency fallback: user message as single search query, all laws.
        fallback_payload = [{"query": user_message, "target_uu": "Semua"}]
        return {
            "search_payloads": fallback_payload,
            "search_filters": {
                "is_chitchat": False,
                "chitchat_response": None,
                "target_uu": "Semua",
            },
        }
