# src/agent/graph.py

from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,  # Impor aio secara presisi
)
from langgraph.graph import END, START, StateGraph

from src.agent.nodes.context import context_manager_node
from src.agent.nodes.input_guard import input_guard_node
from src.agent.nodes.output_guard import output_guard_node
from src.agent.nodes.reasoner import legal_reasoner_node
from src.agent.nodes.search import legal_search_node
from src.agent.state import LegalAgentState
from src.database.connection import async_postgres_pool
from src.logger import logger

# ==========================================
# DEFINISI GERBANG LOGIKA (CONDITIONAL EDGES)
# ==========================================


def route_after_input_guard(state: LegalAgentState):
    if state.get("system_status") == "flagged_input":
        logger.warning("Input berbahaya terdeteksi! Lompat ke Node Akhir.")
        return "end_process"
    return "continue_process"


def route_after_context(state: LegalAgentState):
    filters = state.get("search_filters", {})
    if filters.get("is_chitchat") == True:
        logger.info("Input terdeteksi Chit-Chat. Lompat langsung ke Reasoner.")
        return "skip_search"
    logger.info("Input terdeteksi Pertanyaan Hukum. Lanjut ke Qdrant Search.")
    return "go_to_search"


def route_after_output_guard(state: LegalAgentState):
    return "end_process"


# ==========================================
# 3. CONSTRUCT THE STATE GRAPH
# ==========================================

builder = StateGraph(LegalAgentState)

# Daftarkan semua Node ke dalam Graf
builder.add_node("input_guard", input_guard_node)
builder.add_node("context_manager", context_manager_node)
builder.add_node("legal_search", legal_search_node)
builder.add_node("legal_reasoner", legal_reasoner_node)
builder.add_node("output_guard", output_guard_node)

# Hubungkan Alur dengan Edges & Conditional Edges
builder.add_edge(START, "input_guard")

builder.add_conditional_edges(
    "input_guard",
    route_after_input_guard,
    {"end_process": END, "continue_process": "context_manager"},
)

builder.add_conditional_edges(
    "context_manager",
    route_after_context,
    {"skip_search": "legal_reasoner", "go_to_search": "legal_search"},
)

builder.add_edge("legal_search", "legal_reasoner")
builder.add_edge("legal_reasoner", "output_guard")

builder.add_conditional_edges(
    "output_guard", route_after_output_guard, {"end_process": END}
)

# Sediakan variabel penampung singleton graf global
legal_rag_graph = None


# PERBAIKAN UTAMA: Bungkus kompilasi graf di dalam fungsi agar aman dari siklus impor
def compile_legal_graph():
    """Mengompilasi blueprint graf hukum secara aman menggunakan checkpointer async terverifikasi"""
    global legal_rag_graph
    if legal_rag_graph is not None:
        return legal_rag_graph

    logger.info("Mengompilasi cetak biru LangGraph dengan AsyncPostgresSaver...")

    # Inisialisasi checkpointer dari pool PostgreSQL yang dijamin aktif saat fungsi ini dipanggil
    memory_checkpointer = AsyncPostgresSaver(async_postgres_pool)
    legal_rag_graph = builder.compile(checkpointer=memory_checkpointer)

    logger.info("Cetak biru LangGraph Legal RAG Assistant sukses dikompilasi!")
    return legal_rag_graph
