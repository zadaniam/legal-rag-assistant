"""Pytest configuration and shared fixtures for integration tests."""

import pytest
import pytest_asyncio

from src.database.connection import (
    async_postgres_pool,
    async_qdrant_client,
    close_database_connections,
    init_database_connections,
)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def db_lifespan():
    """Manages connection lifecycle for the entire integration test session."""
    await init_database_connections()
    yield
    await close_database_connections()


@pytest_asyncio.fixture(scope="function")
async def pg_conn(db_lifespan):
    """Provides an isolated Postgres connection per test function."""
    async with async_postgres_pool.connection() as conn:
        yield conn
        await conn.rollback()


@pytest.fixture(scope="function")
def qdrant_cli(db_lifespan):
    """Provides the Qdrant client."""
    return async_qdrant_client
