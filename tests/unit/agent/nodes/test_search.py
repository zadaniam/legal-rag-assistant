from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.agent.nodes.search import legal_search_node
from src.agent.state import LegalAgentState


class FakeLegalRetriever:
    """In-memory test double implementing LegalRetriever protocol."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    async def hybrid_search(
        self,
        search_payloads: list[dict[str, Any]],
        prefetch_limit: int,
        fusion_limit: int,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "search_payloads": search_payloads,
                "prefetch_limit": prefetch_limit,
                "fusion_limit": fusion_limit,
            }
        )
        return self.results


@pytest.mark.anyio
async def test_legal_search_node_delegates_to_injected_retriever() -> None:
    expected_docs = [
        {
            "uu_id": "uu_ketenagakerjaan_2003",
            "pasal": "Pasal 156",
            "text_child": "Komponen pesangon...",
            "text_parent": "Aturan Uang Pesangon",
            "status_keberlakuan": "berlaku",
            "parent_id": "parent-uu-ketenagakerjaan-156",
        },
        {
            "uu_id": "uu_bpjs_2011",
            "pasal": "Pasal 37",
            "text_child": "Klaim JHT...",
            "text_parent": "Aturan Klaim JHT",
            "status_keberlakuan": "berlaku",
            "parent_id": "parent-uu-bpjs-37",
        },
    ]
    fake_retriever = FakeLegalRetriever(results=expected_docs)

    mock_state: LegalAgentState = {
        "messages": [],
        "search_payloads": [
            {
                "query": "aturan uang pesangon",
                "target_uu": "uu_ketenagakerjaan_2003",
            },
            {
                "query": "syarat klaim JHT BPJS",
                "target_uu": "Semua",
            },
        ],
        "search_filters": {},
        "retrieved_documents": [],
        "system_status": "clear",
        "final_answer": "",
    }
    config = {
        "configurable": {
            "retriever": fake_retriever,
            "qdrant_prefetch_limit": 15,
            "qdrant_fusion_limit": 5,
        }
    }

    node_output = await legal_search_node(mock_state, config=config)

    assert node_output == {"retrieved_documents": expected_docs}
    assert len(fake_retriever.calls) == 1
    assert fake_retriever.calls[0]["prefetch_limit"] == 15
    assert fake_retriever.calls[0]["fusion_limit"] == 5
    assert len(fake_retriever.calls[0]["search_payloads"]) == 2


@pytest.mark.anyio
async def test_legal_search_node_skips_when_payloads_empty() -> None:
    mock_retriever = AsyncMock()
    mock_state: LegalAgentState = {
        "messages": [],
        "search_payloads": [],
        "search_filters": {},
        "retrieved_documents": [],
        "system_status": "clear",
        "final_answer": "",
    }

    node_output = await legal_search_node(
        mock_state, config={"configurable": {"retriever": mock_retriever}}
    )

    assert node_output == {"retrieved_documents": []}
    mock_retriever.hybrid_search.assert_not_called()
