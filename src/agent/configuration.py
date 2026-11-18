"""Runtime configuration schema for the Legal RAG LangGraph workflow."""

from dataclasses import dataclass, fields
from typing import Any

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class AgentConfiguration:
    """Configurable runtime parameters for all nodes in the legal agent graph."""

    guard_model: str = "gemini-3.5-flash-lite"
    router_model: str = "gemini-3.5-flash-lite"
    reasoner_model: str = "gemini-3.5-flash-lite"
    auditor_model: str = "gemini-3.5-flash-lite"

    guard_temperature: float = 0.0
    router_temperature: float = 0.0
    reasoner_temperature: float = 0.2
    auditor_temperature: float = 0.0

    qdrant_prefetch_limit: int = 20
    qdrant_fusion_limit: int = 3
    max_history_messages: int = 10

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "AgentConfiguration":
        """Extract valid configuration fields from LangGraph RunnableConfig."""
        configurable = (config or {}).get("configurable", {})
        valid_fields = {f.name for f in fields(cls)}
        filtered_kwargs: dict[str, Any] = {
            k: v for k, v in configurable.items() if k in valid_fields
        }
        return cls(**filtered_kwargs)
