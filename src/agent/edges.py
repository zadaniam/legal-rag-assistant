"""Conditional routing edges for the legal RAG LangGraph workflow."""

from src.agent.state import LegalAgentState
from src.logger import logger


def route_after_input_guard(state: LegalAgentState) -> str:
    """Route to end if input is flagged, otherwise continue to context manager."""
    if state.get("system_status") == "flagged_input":
        logger.warning("Hazardous input detected! Skipping to end node.")
        return "end_process"
    return "continue_process"


def route_after_context(state: LegalAgentState) -> str:
    """Route to reasoner for chitchat, otherwise to search for legal queries."""
    filters = state.get("search_filters", {})
    if filters.get("is_chitchat"):
        logger.info("Input detected as chitchat. Skipping directly to reasoner.")
        return "skip_search"
    logger.info("Input detected as legal query. Proceeding to Qdrant search.")
    return "go_to_search"
