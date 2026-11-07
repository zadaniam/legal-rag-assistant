import pytest
import pytest_asyncio
from src.database.connection import (
    init_database_connections,
    close_database_connections,
    async_postgres_pool,
    async_qdrant_client
)

@pytest_asyncio.fixture(scope="session", autouse=True)
async def db_lifespan():
    """Mengelola siklus hidup koneksi untuk seluruh sesi pengujian integrasi."""
    await init_database_connections()
    yield
    await close_database_connections()

@pytest_asyncio.fixture(scope="function")
async def pg_conn(db_lifespan):
    """Fixture untuk mendapatkan koneksi Postgres yang terisolasi per fungsi tes."""
    async with async_postgres_pool.connection() as conn:
        yield conn
        # agar data sampah dari tes tidak mengotori DB tes untuk tes berikutnya.
        await conn.rollback()

@pytest.fixture(scope="function")
def qdrant_cli(db_lifespan):
    """Fixture untuk mendapatkan client Qdrant."""
    return async_qdrant_client
