"""Pytest configuration and shared fixtures for live E2E API tests."""

import pytest_asyncio

from src.database.connection import (
    close_database_connections,
    get_postgres_pool,
    init_database_connections,
)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def db_lifespan_e2e():
    """Manages the real DB lifecycle for live E2E test sessions."""
    await init_database_connections()
    yield

    pool = get_postgres_pool()
    async with pool.connection() as conn:
        await conn.commit()

    await close_database_connections()
