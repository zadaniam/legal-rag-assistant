"""Legal search node for the legal RAG agent.

Delegates hybrid retrieval to an injected or default LegalRetriever backend
using decomposed queries produced by the context manager.
"""

from typing import Any, cast

from langchain_core.runnables import RunnableConfig

from src.agent.configuration import AgentConfiguration
from src.agent.state import LegalAgentState
from src.database.legal_retriever import LegalRetriever, get_default_legal_retriever
from src.logger import logger


def _resolve_retriever(config: RunnableConfig | None) -> LegalRetriever:
    """Resolve retriever from RunnableConfig if injected, otherwise use default."""
    configurable = (config or {}).get("configurable", {})
    injected = configurable.get("retriever")
    if injected is not None:
        return cast(LegalRetriever, injected)
    return get_default_legal_retriever()


async def legal_search_node(
    state: LegalAgentState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Execute hybrid retrieval using decomposed search payloads."""
    search_payloads = state.get("search_payloads", [])
    if not search_payloads:
        logger.info("No search payloads. Skipping retrieval process.")
        return {"retrieved_documents": []}

    retriever = _resolve_retriever(config)
    agent_cfg = AgentConfiguration.from_runnable_config(config)

    results = await retriever.hybrid_search(
        search_payloads=search_payloads,
        prefetch_limit=agent_cfg.qdrant_prefetch_limit,
        fusion_limit=agent_cfg.qdrant_fusion_limit,
    )
    return {"retrieved_documents": results}
