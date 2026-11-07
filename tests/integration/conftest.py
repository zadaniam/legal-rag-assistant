import pytest_asyncio

from src.database.connection import (
    close_database_connections,
    init_database_connections,
)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def manage_integration_database_lifespan():
    """Hanya berjalan untuk uji integrasi yang membutuhkan database menyala"""

    await init_database_connections()
    yield
    await close_database_connections()
