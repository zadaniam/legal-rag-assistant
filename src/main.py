"""FastAPI application entry point with async lifespan management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.agent.graph import compile_legal_graph
from src.api.v1.chat import router as chat_v1_router
from src.config import settings, setup_telemetry_environment
from src.database.connection import (
    check_database_health,
    close_database_connections,
    init_database_connections,
)
from src.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager handling startup and shutdown."""
    try:
        setup_telemetry_environment()
        logger.info("LangSmith telemetry environment configured successfully.")

        await init_database_connections()
        compile_legal_graph()

        logger.info("Backend agentic RAG async ready to serve")
    except Exception as e:
        logger.error(f"Failed to run application startup cycle: {e}")
        raise

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


@app.get("/health", tags=["Monitoring"])
async def health_check() -> JSONResponse:
    """Verify application and downstream database readiness."""
    db_health: dict[str, Any] = await check_database_health()
    is_healthy = (
        db_health["postgres"]["status"] == "up"
        and db_health["qdrant"]["status"] == "up"
    )

    payload = {
        "status": "healthy" if is_healthy else "degraded",
        "environment": settings.ENVIRONMENT,
        "dependencies": db_health,
    }
    status_code = (
        status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(status_code=status_code, content=payload)


logger.info("FastAPI server initialized with modular APIRouter Lifespan")
