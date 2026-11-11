"""Agent state definition for the legal RAG LangGraph workflow."""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class LegalAgentState(TypedDict):
    """State shared across all nodes in the legal RAG agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    search_payloads: list[dict[str, Any]]
    search_filters: dict[str, Any]
    retrieved_documents: list[dict[str, Any]]
    system_status: str
    final_answer: str
