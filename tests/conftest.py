# tests/conftest.py
import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    """Memaksa anyio/fastapi testing menggunakan backend asyncio"""
    return "asyncio"


@pytest.fixture(scope="session")
def asyncio_default_test_loop_scope():
    """Memastikan scope event loop berjalan pada tingkat session"""
    return "session"
