"""Chat API endpoint for the legal RAG assistant using Server-Sent Events (SSE)."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.logger import logger
from src.schemas.chat import ChatRequest
from src.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Legal Chat Session"])


@router.post("")
async def chat_with_legal_agent(payload: ChatRequest) -> StreamingResponse:
    """Handle incoming chat requests and stream the legal agent response via SSE."""
    logger.info(
        "Received HTTP Chat request via APIRouter SSE Stream. "
        f"Session: {payload.session_id}"
    )

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }

    return StreamingResponse(
        ChatService.stream_chat_response(payload),
        media_type="text/event-stream",
        headers=headers,
    )
