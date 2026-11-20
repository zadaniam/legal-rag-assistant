from typing import cast

from langchain_core.runnables import RunnableConfig

from src.agent.configuration import AgentConfiguration


def test_agent_configuration_defaults_when_config_is_none() -> None:
    cfg = AgentConfiguration.from_runnable_config(None)

    assert cfg.guard_model == "google_genai:gemini-3.5-flash-lite"
    assert cfg.router_model == "google_genai:gemini-3.5-flash-lite"
    assert cfg.reasoner_model == "google_genai:gemini-3.5-flash-lite"
    assert cfg.auditor_model == "google_genai:gemini-3.5-flash-lite"
    assert cfg.reasoner_temperature == 0.2
    assert cfg.qdrant_prefetch_limit == 20
    assert cfg.qdrant_fusion_limit == 3
    assert cfg.max_history_messages == 10


def test_agent_configuration_overrides_and_ignores_unknown_keys() -> None:
    raw_config = cast(
        RunnableConfig,
        {
            "configurable": {
                "thread_id": "session_123",
                "reasoner_model": "gemini-3.5-pro",
                "qdrant_fusion_limit": 5,
                "reasoner_temperature": 0.0,
            }
        },
    )

    cfg = AgentConfiguration.from_runnable_config(raw_config)

    assert cfg.reasoner_model == "gemini-3.5-pro"
    assert cfg.qdrant_fusion_limit == 5
    assert cfg.reasoner_temperature == 0.0
    assert cfg.guard_model == "google_genai:gemini-3.5-flash-lite"
    assert cfg.qdrant_prefetch_limit == 20
