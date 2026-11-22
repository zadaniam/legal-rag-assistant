"""Database connection management for PostgreSQL and Qdrant."""

import asyncio
from typing import Any
from urllib.parse import urlparse

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient

from src.config import settings
from src.logger import logger

async_postgres_pool: AsyncConnectionPool[Any] | None = None
async_qdrant_client: AsyncQdrantClient | None = None


def _resolve_ssl_mode(postgres_url: str) -> str:
    """Determine PostgreSQL SSL mode based on target host."""
    parsed = urlparse(postgres_url)
    hostname = (parsed.hostname or "").lower()
    if hostname in {"localhost", "127.0.0.1", "postgres", ""}:
        return "disable"
    return "require"


def _build_postgres_pool() -> AsyncConnectionPool[Any]:
    """Create a configured async PostgreSQL connection pool without opening it."""
    conninfo = settings.psycopg_postgres_url
    sslmode = _resolve_ssl_mode(conninfo)

    return AsyncConnectionPool(
        open=False,
        conninfo=conninfo,
        min_size=settings.DB_POOL_MIN_SIZE,
        max_size=settings.DB_POOL_MAX_SIZE,
        timeout=settings.DB_POOL_TIMEOUT,
        max_lifetime=settings.DB_POOL_MAX_LIFETIME,
        kwargs={
            "autocommit": True,
            "sslmode": sslmode,
            "row_factory": dict_row,
            "prepare_threshold": None,
        },
    )


def _build_qdrant_client() -> AsyncQdrantClient:
    """Create a configured async Qdrant client supporting HTTP and optional gRPC."""
    api_key = (
        settings.QDRANT_API_KEY.get_secret_value() if settings.QDRANT_API_KEY else None
    )
    return AsyncQdrantClient(
        url=settings.QDRANT_URL,
        grpc_port=settings.QDRANT_GRPC_PORT,
        api_key=api_key,
        timeout=settings.QDRANT_TIMEOUT,
        prefer_grpc=settings.QDRANT_PREFER_GRPC,
    )


def get_postgres_pool() -> AsyncConnectionPool[Any]:
    """Return the active PostgreSQL connection pool or raise if uninitialized."""
    if async_postgres_pool is None:
        raise RuntimeError(
            "PostgreSQL pool is not initialized. "
            "Call 'init_postgres_pool()' or 'init_database_connections()' first."
        )
    return async_postgres_pool


def get_qdrant_client() -> AsyncQdrantClient:
    """Return the active Qdrant client or raise if uninitialized."""
    if async_qdrant_client is None:
        raise RuntimeError(
            "Qdrant client is not initialized. "
            "Call 'init_qdrant_client()' or 'init_database_connections()' first."
        )
    return async_qdrant_client


async def init_postgres_pool() -> AsyncConnectionPool[Any]:
    """Lazily create and open the async PostgreSQL connection pool."""
    global async_postgres_pool
    if async_postgres_pool is None:
        logger.info("Configuring async PostgreSQL connection pool...")
        async_postgres_pool = _build_postgres_pool()

    if async_postgres_pool.closed:
        logger.info("Opening physical async PostgreSQL connection pool...")
        await async_postgres_pool.open()
        await async_postgres_pool.wait(timeout=settings.DB_POOL_TIMEOUT)
        logger.info("Async PostgreSQL connection pool opened and ready.")

    return async_postgres_pool


async def init_qdrant_client() -> AsyncQdrantClient:
    """Lazily create and verify the async Qdrant client."""
    global async_qdrant_client
    if async_qdrant_client is None:
        logger.info(
            f"Configuring async Qdrant client at {settings.QDRANT_URL} "
            f"(prefer_grpc={settings.QDRANT_PREFER_GRPC})..."
        )
        async_qdrant_client = _build_qdrant_client()
        await async_qdrant_client.get_collections()
        logger.info("Async Qdrant client connected and verified.")

    return async_qdrant_client


async def init_database_connections() -> None:
    """Initialize all database connections concurrently during startup."""
    try:
        await asyncio.gather(
            init_postgres_pool(),
            init_qdrant_client(),
        )
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        await close_database_connections()
        raise


async def close_database_connections() -> None:
    """Close all active database connections cleanly during shutdown."""
    global async_postgres_pool, async_qdrant_client

    if async_postgres_pool is not None:
        logger.info("Closing async PostgreSQL connection pool...")
        if not async_postgres_pool.closed:
            await async_postgres_pool.close()
        async_postgres_pool = None
        logger.info("Async PostgreSQL connection pool closed cleanly.")

    if async_qdrant_client is not None:
        logger.info("Closing async Qdrant client...")
        await async_qdrant_client.close()
        async_qdrant_client = None
        logger.info("Async Qdrant client closed cleanly.")


async def check_database_health() -> dict[str, Any]:
    """Ping PostgreSQL and Qdrant to verify live connectivity."""
    health: dict[str, Any] = {
        "postgres": {"status": "down"},
        "qdrant": {"status": "down"},
    }

    try:
        pool = get_postgres_pool()
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT 1 AS ok")
            await cur.fetchone()
        health["postgres"] = {"status": "up"}
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
        health["postgres"] = {"status": "down", "error": str(e)}

    try:
        client = get_qdrant_client()
        await client.get_collections()
        health["qdrant"] = {"status": "up"}
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        health["qdrant"] = {"status": "down", "error": str(e)}

    return health
