# tests/unit/conftest.py
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_infrastructure_lifecycle(session_mocker):
    """Otomatis memotong dan memalsukan seluruh koneksi fisik ke PostgreSQL.
    Mencegah error PoolTimeout selama unit testing.
    """
    mock_pool = MagicMock()
    mock_pool.open = AsyncMock()
    mock_pool.wait = AsyncMock()
    mock_pool.close = AsyncMock()
    mock_pool.connection = AsyncMock()

    mock_saver = MagicMock()
    mock_saver.setup = AsyncMock()

    session_mocker.patch(
        "src.database.connection.AsyncConnectionPool", MagicMock(return_value=mock_pool)
    )
    session_mocker.patch(
        "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver",
        MagicMock(return_value=mock_saver),
    )

    yield
