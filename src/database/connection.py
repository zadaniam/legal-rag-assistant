"""Database connection management for PostgreSQL and Qdrant."""

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient

from src.config import settings
from src.logger import logger

async_postgres_pool: AsyncConnectionPool | None = None
async_qdrant_client: AsyncQdrantClient | None = None


def _configure_postgres_pool() -> AsyncConnectionPool:
    """Create and configure the async PostgreSQL connection pool."""
    postgres_url_str = settings.POSTGRES_URL

    sslmode = "require"
    if "localhost" in postgres_url_str or "127.0.0.1" in postgres_url_str:
        sslmode = "disable"

    return AsyncConnectionPool(
        open=False,
        conninfo=postgres_url_str,
        min_size=1,
        max_size=5,
        kwargs={
            "autocommit": True,
            "sslmode": sslmode,
            "row_factory": dict_row,
        },
    )


def _configure_qdrant_client() -> AsyncQdrantClient:
    qdrant_api_key_str = (
        settings.QDRANT_API_KEY.get_secret_value() if settings.QDRANT_API_KEY else None
    )

    return AsyncQdrantClient(
        url=settings.QDRANT_URL,
        api_key=qdrant_api_key_str,
    )


try:
    logger.info("Initializing async PostgreSQL connection pool...")
    async_postgres_pool = _configure_postgres_pool()
except Exception as e:
    logger.error(f"Failed to configure async PostgreSQL pool: {e}")
    raise

try:
    logger.info(f"Connecting to async Qdrant server at {settings.QDRANT_URL}...")
    async_qdrant_client = _configure_qdrant_client()
    logger.info("Async Qdrant client configured successfully.")
except Exception as e:
    logger.error(f"Failed to connect to async Qdrant client: {e}")
    raise


async def init_database_connections() -> None:
    """Open physical database connections within the active event loop."""
    if async_postgres_pool is None:
        msg = "PostgreSQL pool is not configured"
        raise RuntimeError(msg)
    try:
        logger.info("Opening physical async PostgreSQL connection pool...")
        await async_postgres_pool.open()
        await async_postgres_pool.wait()
        logger.info("Async PostgreSQL connection pool opened and ready.")
    except Exception as e:
        logger.error(f"Failed to open physical async PostgreSQL pool: {e}")
        raise


async def close_database_connections() -> None:
    """Close all database connection pools during application shutdown."""
    if async_postgres_pool is None:
        msg = "PostgreSQL pool is not configured"
        raise RuntimeError(msg)
    logger.info("Closing async PostgreSQL connection pool...")
    await async_postgres_pool.close()
    logger.info("Async PostgreSQL connection pool closed cleanly.")
