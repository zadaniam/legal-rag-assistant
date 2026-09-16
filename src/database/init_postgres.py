import asyncio

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.database.connection import (
    async_postgres_pool,
    close_database_connections,
    init_database_connections,
)
from src.logger import logger


async def init_langgraph_tables():
    """Membuat skema tabel internal LangGraph Checkpointer secara Async"""
    logger.info("Memeriksa tabel internal Async LangGraph Checkpointer...")
    try:
        async with async_postgres_pool.connection() as conn:
            checkpointer = AsyncPostgresSaver(conn)
            await checkpointer.setup()
        logger.info("Tabel LangGraph Checkpointer berhasil diverifikasi (Async).")
    except Exception as e:
        logger.error(f"Gagal menginisialisasi tabel LangGraph: {e}")
        raise e


async def init_postgresql_tables():
    """Membuat tabel bisnis aplikasi di PostgreSQL"""
    logger.info("Memeriksa skema tabel aplikasi di PostgreSQL...")

    create_sessions_table = """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        title VARCHAR(255) DEFAULT 'Diskusi Hukum Ketenagakerjaan',
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

    async with async_postgres_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(create_sessions_table)
        await cur.execute(create_logs_table)
        logger.info("Tabel 'chat_sessions' dan 'legal_audit_logs' aman diverifikasi.")


async def run_postgres_initialization():
    try:
        logger.info("=== MEMULAI INISIALISASI POSTGRESQL ===")
        await init_database_connections()

        await init_postgresql_tables()
        await init_langgraph_tables()

        logger.info("=== POSTGRESQL BERHASIL DIKONFIGURASI ===")
    except Exception as e:
        logger.error(f"Inisialisasi PostgreSQL gagal: {e}")
        raise e
    finally:
        await close_database_connections()


if __name__ == "__main__":
    asyncio.run(run_postgres_initialization())
