# tests/conftest.py
import pytest
import pytest_asyncio

from src.config import settings
from src.database.connection import (
    close_database_connections,
    init_database_connections,
)
from src.logger import logger


@pytest.fixture(scope="session")
def anyio_backend():
    """Memaksa anyio/fastapi testing menggunakan backend asyncio"""
    return "asyncio"


@pytest.fixture(scope="session")
def asyncio_default_test_loop_scope():
    return "session"


@pytest_asyncio.fixture(scope="session", autouse=True)
async def manage_test_database_lifespan():
    """Otomatis membuka kolam koneksi database sebelum tes berjalan dan menutupnya setelah selesai"""

    # KUNCI PERBAIKAN: Jika lingkungan adalah 'testing' (Unit Test/CI), bypass koneksi fisik database
    if settings.ENVIRONMENT == "testing":
        logger.info(
            "⚠️ [CI DETECTED] Menjalankan Unit Test terisolasi. Mem-bypass koneksi fisik database."
        )
        yield
        logger.info("⚠️ [CI DETECTED] Unit Test selesai. Lewati penutupan database.")
        return

    # Jalur Normal (Integration/E2E Lokal dengan Docker menyala)
    logger.info("Opening database connections for Integration/E2E testing...")
    await init_database_connections()

    yield  # Seluruh berkas E2E dan API dijalankan di atas loop yang sama

    # Bersihkan sisa koneksi setelah seluruh tes usai
    logger.info("Closing database connections...")
    await close_database_connections()
