# src/main.py

from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1.chat import router as chat_v1_router
from src.config import settings
from src.database.connection import (
    close_database_connections,
    init_database_connections,
)
from src.logger import logger


# 1. Definisikan Lifespan Manager Aplikasi
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [STARTUP]: Dieksekusi otomatis SAAT server uvicorn dinyalakan
    try:
        # A. Buka koneksi fisik ke Postgres Pool & Qdrant Async
        await init_database_connections()

        # B. Kompilasi blueprint LangGraph di dalam event loop yang aktif
        from src.agent.graph import compile_legal_graph

        compile_legal_graph()

        logger.info("=== BACKEND AGENTIC RAG ASYNC READY TO SERVE ===")
    except Exception as e:
        logger.error(f"Gagal menjalankan siklus startup aplikasi: {e}")
        raise e

    yield  # Aplikasi aktif menerima request HTTP/Stream dari client

    # [SHUTDOWN]: Dieksekusi otomatis SAAT server uvicorn dimatikan
    await close_database_connections()


# 2. Inisialisasi FastAPI dengan Lifespan
app = FastAPI(
    title="Asisten Legal RAG API",
    description="API HTTP Clean Architecture dengan Lifespan Async untuk Agentic RAG Hukum",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_v1_router, prefix="/api/v1")


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Monitoring"])
def health_check():
    return {"status": "healthy", "environment": settings.ENVIRONMENT}


logger.info("FastAPI Server sukses membaca arsitektur APIRouter modular Lifespan.")
