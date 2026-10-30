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
            # 1. Jalankan stream murni hanya untuk memberikan sinyal UI loading/node aktif
            async for chunk in legal_rag_graph.astream(
                input_state, config=config, stream_mode="updates"
            ):
                chunk_keys = list(chunk.keys())
                if len(chunk_keys) == 0:
                    continue

                node_name = chunk_keys[0]

                # Kirim sinyal node aktif ke Chainlit untuk visual progress bar
                payload_node = {"event": "node_complete", "node": node_name}
                yield f"data: {json.dumps(payload_node)}\n\n"

            # 2. SOLUSI UTAMA: Ambil snapshot state TERAKHIR yang 100% utuh & valid dari LangGraph
            graph_state = await legal_rag_graph.aget_state(config)
            final_state = (
                graph_state.values
            )  # Ini adalah dictionary berisi state akhir yang komplit

            # 3. Penyusunan payload jawaban final menggunakan data dari final_state yang terjamin
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
                "final_answer": final_state.get(
                    "final_answer", ""
                ),  # Dijamin tidak akan kosong lagi
                "system_status": final_state.get("system_status", "clear"),
                "is_chitchat": is_chitchat,
                "citations": citations_output,
            }

            logger.info(
                f"Streaming selesai. Mengirimkan final_result untuk sesi: {payload.session_id}"
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
            f"Gagal memproses HTTP Chat SSE pada sesi {payload.session_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kendala teknis internal pada mesin AI: {e!s}",
        ) from e
