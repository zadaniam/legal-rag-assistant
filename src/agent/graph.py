"""LangGraph blueprint for the legal RAG agent workflow."""

from typing import Any, cast

from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.nodes.context import context_manager_node
from src.agent.nodes.input_guard import input_guard_node
from src.agent.nodes.output_guard import output_guard_node
from src.agent.nodes.reasoner import legal_reasoner_node
from src.agent.nodes.search import legal_search_node
from src.agent.state import LegalAgentState
from src.database.connection import async_postgres_pool
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


def route_after_output_guard(state: LegalAgentState) -> str:
    """Always route to end after output guard."""
    return "end_process"


builder = StateGraph(LegalAgentState)

builder.add_node("input_guard", input_guard_node)
builder.add_node("context_manager", context_manager_node)
builder.add_node("legal_search", legal_search_node)
builder.add_node("legal_reasoner", legal_reasoner_node)
builder.add_node("output_guard", output_guard_node)

builder.add_edge(START, "input_guard")

builder.add_conditional_edges(
    "input_guard",
    route_after_input_guard,
    {"end_process": END, "continue_process": "context_manager"},
)

builder.add_conditional_edges(
    "context_manager",
    route_after_context,
    {"skip_search": "legal_reasoner", "go_to_search": "legal_search"},
)

builder.add_edge("legal_search", "legal_reasoner")
builder.add_edge("legal_reasoner", "output_guard")

builder.add_conditional_edges(
    "output_guard", route_after_output_guard, {"end_process": END}
)

legal_rag_graph: CompiledStateGraph | None = None


def compile_legal_graph() -> CompiledStateGraph:
    """Compile the LangGraph blueprint with an async Postgres checkpointer.

    The checkpointer is initialized from the PostgreSQL connection pool, which
    is guaranteed to be open when this function is called during application
    startup. Wrapping compilation in a function avoids import-cycle issues.
    """
    global legal_rag_graph
    if legal_rag_graph is not None:
        return legal_rag_graph

    logger.info("Compiling LangGraph blueprint with AsyncPostgresSaver...")

    if async_postgres_pool is None:
        raise RuntimeError(
            "Postgres pool is not initialized. Ensure connection is active."
        )

    postgres_pool = cast(Any, async_postgres_pool)
    memory_checkpointer = AsyncPostgresSaver(postgres_pool)

    legal_rag_graph = builder.compile(checkpointer=memory_checkpointer)

    logger.info("Legal RAG Assistant LangGraph blueprint compiled successfully")
    return legal_rag_graph
