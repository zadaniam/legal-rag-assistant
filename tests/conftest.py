# tests/conftest.py
import pytest
import pytest_asyncio

from src.database.connection import (
    close_database_connections,
    init_database_connections,
)


@pytest.fixture(scope="session")
def anyio_backend():
    """Memaksa anyio/fastapi testing menggunakan backend asyncio"""
    return "asyncio"


# PERBAIKAN CORE: Paksa pytest-asyncio agar membagikan satu event loop yang sama sepanjang sesi
@pytest.fixture(scope="session")
def asyncio_default_test_loop_scope():
    return "session"


@pytest_asyncio.fixture(scope="session", autouse=True)
async def manage_test_database_lifespan():
    """Otomatis membuka kolam koneksi database sebelum tes berjalan dan menutupnya setelah selesai"""
    # Buka koneksi fisik ke Postgres & Qdrant secara aman di event loop tunggal
    await init_database_connections()

    yield  # Seluruh berkas E2E dan API dijalankan di atas loop yang sama

    # Bersihkan sisa koneksi setelah seluruh tes usai
    await close_database_connections()
