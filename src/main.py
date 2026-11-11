"""FastAPI application entry point with async lifespan management."""

from collections.abc import AsyncGenerator
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager handling startup and shutdown."""
    try:
        from src.config import setup_telemetry_environment

        setup_telemetry_environment()
        logger.info("LangSmith telemetry environment configured successfully.")

        await init_database_connections()

        from src.agent.graph import compile_legal_graph

        compile_legal_graph()

        logger.info("Backend agentic RAG async ready to serve")
    except Exception as e:
        logger.error(f"Failed to run application startup cycle: {e}")
        raise e

    yield

    await close_database_connections()


app = FastAPI(
    title="Legal RAG Assistant API",
    description=(
        "HTTP Clean Architecture API with Async Lifespan for Agentic Legal RAG"
    ),
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
def health_check() -> dict[str, str]:
    """Return application health status and current environment."""
    return {"status": "healthy", "environment": settings.ENVIRONMENT}


logger.info("FastAPI server initialized with modular APIRouter Lifespan")
