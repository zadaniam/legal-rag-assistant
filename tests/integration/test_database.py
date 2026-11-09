import pytest


@pytest.mark.asyncio
async def test_postgres_connection_query(pg_conn):
    """Memastikan kolam koneksi Postgres terbuka dan bisa mengeksekusi kueri dasar"""
    async with pg_conn.cursor() as cur:
        await cur.execute("SELECT 1 AS total;")
        result = await cur.fetchone()

    assert result == {"total": 1}


@pytest.mark.asyncio
async def test_qdrant_connection_ping(qdrant_cli):
    """Memastikan client Qdrant terhubung dan bisa membaca daftar koleksi"""
    response = await qdrant_cli.get_collections()

    assert response is not None
    assert hasattr(response, "collections")
