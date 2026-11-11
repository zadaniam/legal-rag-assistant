"""Chat API endpoint for the legal RAG assistant using Server-Sent Events (SSE)."""

import json
from collections.abc import AsyncGenerator
from typing import cast

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.agent.graph import compile_legal_graph
from src.logger import logger

router = APIRouter(prefix="/chat", tags=["Legal Chat Session"])


class ChatRequest(BaseModel):
    """Request payload for the legal chat endpoint."""

    message: str = Field(..., description="Legal question or greeting from the user.")
    session_id: str = Field(
        ...,
        description=(
            "Unique chat session ID (thread_id) for Postgres conversation memory."
        ),
    )


@router.post("")
async def chat_with_legal_agent(payload: ChatRequest) -> StreamingResponse:
    """Handle incoming chat requests and stream the legal agent response via SSE."""
    logger.info(
        "Received HTTP Chat request via APIRouter SSE Stream. "
        f"Session: {payload.session_id}"
    )

    input_state = {"messages": [HumanMessage(content=payload.message)]}
    config = cast(RunnableConfig, {"configurable": {"thread_id": payload.session_id}})

    try:
        legal_rag_graph = compile_legal_graph()

        async def event_generator() -> AsyncGenerator[str, None]:
            async for chunk in legal_rag_graph.astream(
                input_state, config=config, stream_mode="updates"
            ):
                chunk_keys = list(chunk.keys())
                if len(chunk_keys) == 0:
                    continue

                node_name = chunk_keys[0]

                payload_node = {"event": "node_complete", "node": node_name}
                yield f"data: {json.dumps(payload_node)}\n\n"

            graph_state = await legal_rag_graph.aget_state(config)
            final_state = graph_state.values

            filters = final_state.get("search_filters", {}) or {}
            is_chitchat = filters.get("is_chitchat", False)

            citations_output = []
            retrieved_docs = final_state.get("retrieved_documents", []) or []
            for doc in retrieved_docs:
                citations_output.append(
                    {
                        "uu_id": doc.get("uu_id", "UU"),
                        "bab": doc.get("bab", "N/A"),
                        "pasal": str(doc.get("pasal", "")),
                        "text_child": doc.get("text_child", ""),
                        "text_parent": doc.get("text_parent", ""),
                    }
                )

            payload_final = {
                "event": "final_result",
                "session_id": payload.session_id,
                "final_answer": final_state.get("final_answer", ""),
                "system_status": final_state.get("system_status", "clear"),
                "is_chitchat": is_chitchat,
                "citations": citations_output,
            }

            logger.info(
                "Streaming complete. Sending final_result "
                f"for session: {payload.session_id}"
            )
            yield f"data: {json.dumps(payload_final)}\n\n"

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
        return StreamingResponse(
            event_generator(), media_type="text/event-stream", headers=headers
        )

    except Exception as e:
        logger.exception(
            f"Failed to process HTTP Chat SSE on session {payload.session_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kendala teknis internal pada mesin AI: {e!s}",
        ) from e
