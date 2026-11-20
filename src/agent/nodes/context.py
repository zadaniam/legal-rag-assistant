"""Context manager node for the legal RAG agent.

Routes user input to either chitchat response or legal search based on
semantic classification using a provider-agnostic chat model.
"""

from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.agent.configuration import AgentConfiguration
from src.agent.prompts import CONTEXT_MANAGER_SYSTEM_PROMPT
from src.agent.state import LegalAgentState
from src.logger import logger
from src.utils.llm import get_chat_model


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
    """Schema for LLM router output classifying user input."""

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


async def context_manager_node(
    state: LegalAgentState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Classify user input and extract search payloads using structured LLM router."""
    all_messages = state.get("messages", [])
    agent_cfg = AgentConfiguration.from_runnable_config(config)

    history_limit = agent_cfg.max_history_messages
    recent_history_messages = (
        all_messages[:-1][-history_limit:] if len(all_messages) > 1 else []
    )

    formatted_recent_history = ""
    for msg in recent_history_messages:
        if isinstance(msg, HumanMessage):
            formatted_recent_history += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            formatted_recent_history += f"Asisten: {msg.content}\n"

    user_message = str(all_messages[-1].content)

    full_system_instruction = CONTEXT_MANAGER_SYSTEM_PROMPT
    if formatted_recent_history:
        full_system_instruction += (
            f"\n\n[Konteks Tambahan] 5 Turn Percakapan Terakhir:\n"
            f"{formatted_recent_history}"
        )

    try:
        llm = get_chat_model(
            model=agent_cfg.router_model,
            temperature=agent_cfg.router_temperature,
        )
        structured_llm = llm.with_structured_output(RouterOutputSchema)

        messages = [
            SystemMessage(content=full_system_instruction),
            HumanMessage(content=user_message),
        ]
        result = cast(RouterOutputSchema, await structured_llm.ainvoke(messages))
        result_dict = result.model_dump()

        logger.info(
            f"LLM Router processed route successfully. is_chitchat={result.is_chitchat}"
        )

        return {
            "search_payloads": result_dict.get("search_payloads", []),
            "search_filters": result_dict,
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
