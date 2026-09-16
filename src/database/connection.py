# src/database/connection.py
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient

from src.config import settings
from src.logger import logger

# 1. Definisikan objek secara pasif (JANGAN langsung dibuka saat import)
try:
    logger.info("Menginisialisasi objek konfigurasi ASYNC PostgreSQL...")

    # Ambil nilai rahasia dari Pydantic Settings
    postgres_url_str = settings.POSTGRES_URL

    async_postgres_pool = AsyncConnectionPool(
        open=False,
        conninfo=postgres_url_str,
        min_size=1,  # PERUBAHAN: Ubah ke 1 agar ramah cold-start serverless
        max_size=5,  # PERUBAHAN: Jangan terlalu besar untuk tier gratis database cloud
        kwargs={
            "autocommit": True,
            # PENYESUAIAN SSL: Memastikan koneksi aman diterima oleh Neon/Supabase
            "sslmode": "require"
            if "localhost" not in postgres_url_str
            and "127.0.0.1" not in postgres_url_str
            else "disable",
        },
    )
except Exception as e:
    logger.error(f"Gagal mengonfigurasi objek Async PostgreSQL: {e}")
    raise e

# 2. Inisialisasi Async Qdrant Client
try:
    logger.info(f"Menghubungkan ke Async Qdrant Server di {settings.QDRANT_URL}...")

    # PENYESUAIAN QDRANT: Mengekstrak string token jika ada (untuk Qdrant Cloud)
    qdrant_api_key_str = (
        settings.QDRANT_API_KEY.get_secret_value() if settings.QDRANT_API_KEY else None
    )

    async_qdrant_client = AsyncQdrantClient(
        url=settings.QDRANT_URL,
        api_key=qdrant_api_key_str,  # Akan otomatis diabaikan oleh Qdrant jika nilainya None (Lokal Docker)
    )
    logger.info("Koneksi ke Async Qdrant Client berhasil dikonfigurasi.")
except Exception as e:
    logger.error(f"Gagal menghubungkan ke Async Qdrant Client: {e}")
    raise e


# 3. FUNGSI MANAJER DAUR HIDUP (Lifespan) untuk mengaktifkan koneksi fisik
async def init_database_connections():
    """Membuka kolam koneksi database secara aman di dalam event loop yang aktif"""
    try:
        logger.info("Membuka kolam koneksi fisik Async ke PostgreSQL...")
        await async_postgres_pool.open()  # Membuka koneksi secara eksplisit dan aman
        await async_postgres_pool.wait()
        logger.info(
            "Kolam koneksi Async PostgreSQL berhasil dibuka dan siap digunakan."
        )
    except Exception as e:
        logger.error(f"Gagal membuka kolam koneksi fisik PostgreSQL: {e}")
        raise e


async def close_database_connections():
    """Menutup seluruh kolam koneksi database saat aplikasi dimatikan"""
    logger.info("Menutup kolam koneksi Async PostgreSQL...")
    await async_postgres_pool.close()
    logger.info("Kolam koneksi Async PostgreSQL berhasil ditutup bersih.")
