# src/agent/state.py

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class LegalAgentState(TypedDict):
    # Menyimpan riwayat pesan obrolan (efisien, dikelola secara otomatis)
    messages: Annotated[list[BaseMessage], add_messages]

    # Payload query hasil ekstraksi Node 2 untuk kebutuhan Multi-UU di Node 3
    search_payloads: list[dict[str, Any]]

    # Mendaftarkan filter pencarian & metadata chitchat
    search_filters: dict[str, Any]

    # Tempat menampung teks pasal-pasal hukum utuh (Parent Chunk) hasil retrieval Node 3
    retrieved_documents: list[dict[str, Any]]

    # Status internal sistem untuk kendali Guardrails (misal: "clear", "flagged_input", "flagged_output")
    system_status: str

    # Jawaban akhir yang sudah lolos kurasi ketat untuk dikirim ke user
    final_answer: str
