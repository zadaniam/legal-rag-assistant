"""Pytest configuration and shared fixtures for the test suite."""

import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    """Force anyio/fastapi testing to use the asyncio backend."""
    return "asyncio"


@pytest.fixture(scope="session")
def asyncio_default_test_loop_scope():
    """Ensure the event loop scope runs at session level."""
    return "session"
