"""Pytest configuration and shared fixtures for live E2E API tests."""

import pytest_asyncio

from src.database.connection import (
    async_postgres_pool,
    close_database_connections,
    init_database_connections,
)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def db_lifespan_e2e():
    """Manages the real DB lifecycle for live E2E test sessions."""
    await init_database_connections()
    yield

    async with async_postgres_pool.connection() as conn:
        await conn.commit()

    await close_database_connections()
