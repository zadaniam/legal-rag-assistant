# tests/unit/test_database.py
import pytest

from src.database.connection import async_postgres_pool, async_qdrant_client


@pytest.mark.asyncio
async def test_postgres_connection_query():
    """Memastikan kolam koneksi Postgres terbuka dan bisa mengeksekusi kueri dasar"""
    # Mengambil satu koneksi dari pool secara async
    async with async_postgres_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("SELECT 1;")
        result = await cur.fetchone()

    assert result == (1,)


@pytest.mark.asyncio
async def test_qdrant_connection_ping():
    """Memastikan client Qdrant terhubung dan bisa membaca daftar koleksi"""
    # Menembak server Qdrant untuk mengambil nama-nama koleksi aktif
    response = await async_qdrant_client.get_collections()

    assert response is not None
    assert hasattr(response, "collections")
