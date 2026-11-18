from langchain_core.messages import HumanMessage

from src.agent.edges import route_after_context, route_after_input_guard
from src.agent.state import LegalAgentState


def _build_base_state(**overrides) -> LegalAgentState:
    state: LegalAgentState = {
        "messages": [HumanMessage(content="Apa syarat kepesertaan BPJS Kesehatan?")],
        "search_payloads": [],
        "search_filters": {},
        "retrieved_documents": [],
        "system_status": "clear",
        "final_answer": "",
    }
    state.update(overrides)
    return state


def test_route_after_input_guard_flagged_returns_end_process() -> None:
    state = _build_base_state(system_status="flagged_input")
    assert route_after_input_guard(state) == "end_process"


def test_route_after_input_guard_clear_returns_continue_process() -> None:
    state = _build_base_state(system_status="clear")
    assert route_after_input_guard(state) == "continue_process"


def test_route_after_context_chitchat_returns_skip_search() -> None:
    state = _build_base_state(search_filters={"is_chitchat": True})
    assert route_after_context(state) == "skip_search"


def test_route_after_context_legal_query_returns_go_to_search() -> None:
    state = _build_base_state(
        search_filters={"is_chitchat": False, "target_uu": "uu_bpjs_2011"}
    )
    assert route_after_context(state) == "go_to_search"


def test_route_after_context_empty_filters_defaults_to_go_to_search() -> None:
    state = _build_base_state(search_filters={})
    assert route_after_context(state) == "go_to_search"
