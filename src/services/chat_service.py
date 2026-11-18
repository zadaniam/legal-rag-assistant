"""Service layer orchestrating LangGraph execution and SSE event formatting."""

import json
from collections.abc import AsyncGenerator
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.agent.graph import compile_legal_graph
from src.agent.state import LegalAgentState
from src.logger import logger
from src.schemas.chat import ChatFinalResponse, ChatRequest, CitationResponse


class ChatService:
    """Service for handling legal RAG agent conversations."""

    @staticmethod
    async def stream_chat_response(payload: ChatRequest) -> AsyncGenerator[str, None]:
        """Execute the legal RAG graph and yield formatted SSE events."""
        input_state = cast(
            LegalAgentState, {"messages": [HumanMessage(content=payload.message)]}
        )
        config = cast(
            RunnableConfig, {"configurable": {"thread_id": payload.session_id}}
        )

        try:
            legal_rag_graph = compile_legal_graph()

            async for chunk in legal_rag_graph.astream(
                input_state, config=config, stream_mode="updates"
            ):
                chunk_keys = list(chunk.keys())
                if not chunk_keys:
                    continue

                node_name = chunk_keys[0]
                payload_node = {"event": "node_complete", "node": node_name}
                yield f"data: {json.dumps(payload_node)}\n\n"

            graph_state = await legal_rag_graph.aget_state(config)
            final_state = graph_state.values

            filters = final_state.get("search_filters", {}) or {}
            is_chitchat = bool(filters.get("is_chitchat", False))

            retrieved_docs = final_state.get("retrieved_documents", []) or []
            citations_output = [
                CitationResponse(
                    uu_id=doc.get("uu_id", "UU"),
                    bab=doc.get("bab", "N/A"),
                    pasal=str(doc.get("pasal", "")),
                    text_child=doc.get("text_child", ""),
                    text_parent=doc.get("text_parent", ""),
                )
                for doc in retrieved_docs
            ]

            final_response = ChatFinalResponse(
                session_id=payload.session_id,
                final_answer=final_state.get("final_answer", ""),
                system_status=final_state.get("system_status", "clear"),
                is_chitchat=is_chitchat,
                citations=citations_output,
            )

            logger.info(
                "Streaming complete. Sending final_result "
                f"for session: {payload.session_id}"
            )
            yield f"data: {json.dumps(final_response.model_dump())}\n\n"

        except Exception as e:
            logger.exception(
                f"Error during SSE stream execution on session "
                f"{payload.session_id}: {e}"
            )
            error_payload = {
                "event": "error",
                "detail": f"Terjadi kendala teknis internal pada mesin AI: {e!s}",
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
