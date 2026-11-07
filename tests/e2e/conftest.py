# tests/e2e/conftest.py
import pytest_asyncio

from src.database.connection import (
    async_postgres_pool,
    close_database_connections,
    init_database_connections,
)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def db_lifespan_e2e():
    """Mengelola siklus hidup DB asli khusus untuk sesi pengujian live E2E."""
    await init_database_connections()
    yield

    # Hapus data log chat kotor dari HTTP Request di Postgres
    async with async_postgres_pool.connection() as conn:
        await conn.commit()

    await close_database_connections()
