# src/api/v1/chat.py
import json

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.agent.graph import compile_legal_graph
from src.logger import logger

router = APIRouter(prefix="/chat", tags=["Chat Sesi Hukum"])


class ChatRequest(BaseModel):
    message: str = Field(
        ..., description="Pesan pertanyaan hukum atau sapaan dari user."
    )
    session_id: str = Field(
        ...,
        description="ID unik sesi obrolan (thread_id) untuk database memori Postgres.",
    )


@router.post("")
async def chat_with_legal_agent(payload: ChatRequest):
    logger.info(
        f"Menerima request HTTP Chat via APIRouter SSE Stream. Sesi: {payload.session_id}"
    )

    input_state = {"messages": [HumanMessage(content=payload.message)]}
    config = {"configurable": {"thread_id": payload.session_id}}

    try:
        legal_rag_graph = compile_legal_graph()

        async def event_generator():
            accumulated_state = {}

            # Membaca event updates murni dari LangGraph
            async for chunk in legal_rag_graph.astream(
                input_state, config=config, stream_mode="updates"
            ):
                # Pengaman jika chunk kosong agar tidak IndexError
                chunk_keys = list(chunk.keys())
                if len(chunk_keys) == 0:
                    continue

                node_name = chunk_keys[0]
                node_data = chunk[node_name]

                # Akumulasikan state
                accumulated_state.update(node_data)
                if "retrieved_documents" in node_data:
                    accumulated_state["retrieved_documents"] = node_data[
                        "retrieved_documents"
                    ]
                if "search_filters" in node_data:
                    accumulated_state["search_filters"] = node_data["search_filters"]

                # Kirim sinyal node aktif ke Chainlit
                payload_node = {"event": "node_complete", "node": node_name}
                yield f"data: {json.dumps(payload_node)}\n\n"

            # Penyusunan payload jawaban final setelah semua node (termasuk output_guard) selesai
            filters = accumulated_state.get("search_filters", {}) or {}
            is_chitchat = filters.get("is_chitchat", False)

            citations_output = []
            retrieved_docs = accumulated_state.get("retrieved_documents", []) or []
            for doc in retrieved_docs:
                citations_output.append(
                    {
                        "uu_id": doc.get("uu_id", "UU"),
                        "pasal": str(doc.get("pasal", "")),
                        "ayat": str(doc.get("ayat", "1")),
                        "text_child": doc.get("text_child", ""),
                    }
                )

            payload_final = {
                "event": "final_result",
                "session_id": payload.session_id,
                "final_answer": accumulated_state.get("final_answer", ""),
                "system_status": accumulated_state.get("system_status", "clear"),
                "is_chitchat": is_chitchat,
                "citations": citations_output,
            }
            # Kirim jawaban utuh yang 100% aman
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
            f"Gagal memproses HTTP Chat SSE pada sesi {payload.session_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kendala teknis internal pada mesin AI: {e!s}",
        )
