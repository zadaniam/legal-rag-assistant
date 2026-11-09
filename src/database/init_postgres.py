"""PostgreSQL database schema initialization."""

import asyncio
from typing import Any, cast

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from src.database.connection import (
    async_postgres_pool,
    close_database_connections,
    init_database_connections,
)
from src.logger import logger


async def init_langgraph_tables() -> None:
    """Create LangGraph Checkpointer internal tables asynchronously."""
    logger.info("Checking async LangGraph Checkpointer internal tables...")
    try:
        if async_postgres_pool is None:
            msg = "PostgreSQL pool is not configured"
            raise RuntimeError(msg)
        typed_pool = cast(
            AsyncConnectionPool[Any],
            async_postgres_pool,
        )
        checkpointer = AsyncPostgresSaver(typed_pool)
        await checkpointer.setup()
        logger.info("LangGraph Checkpointer tables verified (async).")
    except Exception as e:
        logger.error(f"Failed to initialize LangGraph tables: {e}")
        raise


async def init_postgresql_tables() -> None:
    """Create application business tables in PostgreSQL."""
    logger.info("Checking application table schema in PostgreSQL...")

    create_sessions_table = """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        title VARCHAR(255) DEFAULT 'Legal Employment Discussion',
        summary_memory TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    create_logs_table = """
    CREATE TABLE IF NOT EXISTS legal_audit_logs (
        log_id SERIAL PRIMARY KEY,
        session_id VARCHAR(255) REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
        user_query TEXT NOT NULL,
        is_safe_input BOOLEAN,
        detected_uu_targets JSONB,
        retrieved_pasal_ids JSONB,
        final_ai_response TEXT,
        is_hallucinated BOOLEAN DEFAULT FALSE,
        latency_ms INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    if async_postgres_pool is None:
        msg = "PostgreSQL pool is not configured"
        raise RuntimeError(msg)
    async with async_postgres_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(create_sessions_table)
        await cur.execute(create_logs_table)
        logger.info("Tables 'chat_sessions' and 'legal_audit_logs' verified.")


async def run_postgres_initialization() -> None:
    """Run full PostgreSQL initialization sequence."""
    try:
        logger.info("Starting PostgreSQL initialization...")
        await init_database_connections()

        await init_postgresql_tables()
        await init_langgraph_tables()

        logger.info("PostgreSQL configured successfully.")
    except Exception as e:
        logger.error(f"PostgreSQL initialization failed: {e}")
        raise
    finally:
        await close_database_connections()


if __name__ == "__main__":
    asyncio.run(run_postgres_initialization())
