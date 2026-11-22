"""Unit tests for database connection lifecycle and configuration normalization."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Settings
from src.database import connection


@pytest.fixture(autouse=True)
def reset_connection_globals():
    """Ensure module-level connection globals are clean before and after each test."""
    connection.async_postgres_pool = None
    connection.async_qdrant_client = None
    yield
    connection.async_postgres_pool = None
    connection.async_qdrant_client = None


def test_psycopg_postgres_url_normalization() -> None:
    """Verify SQLAlchemy-style Postgres URL is normalized for libpq."""
    cfg_sqlalchemy = Settings(
        GEMINI_API_KEY="test-key",
        LANGSMITH_API_KEY="test-key",
        POSTGRES_URL="postgresql+psycopg://user:pass@localhost:5432/db",
    )
    assert (
        cfg_sqlalchemy.psycopg_postgres_url
        == "postgresql://user:pass@localhost:5432/db"
    )

    cfg_standard = Settings(
        GEMINI_API_KEY="test-key",
        LANGSMITH_API_KEY="test-key",
        POSTGRES_URL="postgresql://user:pass@localhost:5432/db",
    )
    assert (
        cfg_standard.psycopg_postgres_url == "postgresql://user:pass@localhost:5432/db"
    )


@pytest.mark.parametrize(
    ("url", "expected_ssl"),
    [
        ("postgresql://user:pass@localhost:5432/db", "disable"),
        ("postgresql://user:pass@127.0.0.1:5432/db", "disable"),
        ("postgresql://user:pass@postgres:5432/db", "disable"),
        ("postgresql://user:pass@aws.com", "require"),
    ],
)
def test_resolve_ssl_mode(url: str, expected_ssl: str) -> None:
    """Verify SSL mode resolves to disable for local hosts and require for remote."""
    assert connection._resolve_ssl_mode(url) == expected_ssl


def test_getters_raise_when_uninitialized() -> None:
    """Verify getters raise RuntimeError before connections are initialized."""
    with pytest.raises(RuntimeError, match="PostgreSQL pool is not initialized"):
        connection.get_postgres_pool()

    with pytest.raises(RuntimeError, match="Qdrant client is not initialized"):
        connection.get_qdrant_client()


@pytest.mark.asyncio
async def test_init_and_close_database_connections_success() -> None:
    """Verify concurrent initialization and idempotent shutdown of both clients."""
    mock_pool = MagicMock()
    mock_pool.closed = True
    mock_pool.open = AsyncMock(side_effect=lambda: setattr(mock_pool, "closed", False))
    mock_pool.wait = AsyncMock()
    mock_pool.close = AsyncMock(side_effect=lambda: setattr(mock_pool, "closed", True))

    mock_qdrant = MagicMock()
    mock_qdrant.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    mock_qdrant.close = AsyncMock()

    with (
        patch("src.database.connection._build_postgres_pool", return_value=mock_pool),
        patch("src.database.connection._build_qdrant_client", return_value=mock_qdrant),
    ):
        await connection.init_database_connections()

        assert connection.get_postgres_pool() is mock_pool
        assert connection.get_qdrant_client() is mock_qdrant
        mock_pool.open.assert_awaited_once()
        mock_pool.wait.assert_awaited_once()
        mock_qdrant.get_collections.assert_awaited_once()

        await connection.close_database_connections()
        mock_pool.close.assert_awaited_once()
        mock_qdrant.close.assert_awaited_once()

        await connection.close_database_connections()


@pytest.mark.asyncio
async def test_init_database_connections_rolls_back_on_partial_failure() -> None:
    """Verify Postgres pool is closed if Qdrant fails during startup."""
    mock_pool = MagicMock()
    mock_pool.closed = True
    mock_pool.open = AsyncMock(side_effect=lambda: setattr(mock_pool, "closed", False))
    mock_pool.wait = AsyncMock()
    mock_pool.close = AsyncMock(side_effect=lambda: setattr(mock_pool, "closed", True))

    mock_qdrant = MagicMock()
    mock_qdrant.get_collections = AsyncMock(side_effect=ConnectionError("Qdrant down"))
    mock_qdrant.close = AsyncMock()

    with (
        patch("src.database.connection._build_postgres_pool", return_value=mock_pool),
        patch("src.database.connection._build_qdrant_client", return_value=mock_qdrant),
        pytest.raises(ConnectionError, match="Qdrant down"),
    ):
        await connection.init_database_connections()

    mock_pool.close.assert_awaited_once()
    mock_qdrant.close.assert_awaited_once()
    assert connection.async_postgres_pool is None
    assert connection.async_qdrant_client is None


@pytest.mark.asyncio
async def test_check_database_health_states() -> None:
    """Verify health check reports both healthy (up) and degraded (down) states."""
    # 1. Both down when uninitialized
    down_health = await connection.check_database_health()
    assert down_health["postgres"]["status"] == "down"
    assert down_health["qdrant"]["status"] == "down"

    # 2. Both up when active
    mock_cursor = AsyncMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__aexit__ = AsyncMock(return_value=None)

    mock_pool = MagicMock()
    mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)

    mock_qdrant = MagicMock()
    mock_qdrant.get_collections = AsyncMock(return_value=MagicMock(collections=[]))

    connection.async_postgres_pool = mock_pool
    connection.async_qdrant_client = mock_qdrant

    up_health = await connection.check_database_health()
    assert up_health == {
        "postgres": {"status": "up"},
        "qdrant": {"status": "up"},
    }
