"""Pydantic schemas for the Legal Chat API endpoints and SSE events."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request payload for the legal chat endpoint."""

    message: str = Field(..., description="Legal question or greeting from the user.")
    session_id: str = Field(
        ...,
        description=(
            "Unique chat session ID (thread_id) for Postgres conversation memory."
        ),
    )


class CitationResponse(BaseModel):
    """Schema for a single legal document citation."""

    uu_id: str = Field(default="UU")
    bab: str = Field(default="N/A")
    pasal: str = Field(default="")
    text_child: str = Field(default="")
    text_parent: str = Field(default="")


class ChatFinalResponse(BaseModel):
    """Schema for the final SSE payload sent when the agent completes."""

    event: str = Field(default="final_result")
    session_id: str
    final_answer: str = Field(default="")
    system_status: str = Field(default="clear")
    is_chitchat: bool = Field(default=False)
    citations: list[CitationResponse] = Field(default_factory=list)
