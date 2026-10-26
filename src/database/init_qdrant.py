# src/database/init_qdrant.py

import asyncio

from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    SparseVectorParams,
    TextIndexParams,
    TokenizerType,
    VectorParams,
)

from src.config import settings
from src.database.connection import async_qdrant_client
from src.logger import logger


async def init_qdrant_collections():
    collection_name = settings.QDRANT_COLLECTION_NAME
    logger.info(
        f"Memeriksa keberadaan koleksi '{collection_name}' di Qdrant Async ({settings.ENVIRONMENT.upper()})..."
    )

    collections_response = await async_qdrant_client.get_collections()
    existing_collections = [c.name for c in collections_response.collections]

    if collection_name in existing_collections:
        if settings.ENVIRONMENT.lower() in ["dev", "development", "local"]:
            logger.warning(
                f"Koleksi lama '{collection_name}' terdeteksi di mode DEV. Menghapus untuk migrasi ke Hybrid Search..."
            )
            try:
                await async_qdrant_client.delete_collection(
                    collection_name=collection_name
                )
                logger.info(f"Koleksi lama '{collection_name}' berhasil dihapus.")
            except Exception as e:
                logger.error(f"Gagal menghapus koleksi lama: {e}")
                raise e
        else:
            logger.error(
                f"BERBAHAYA: Koleksi '{collection_name}' sudah ada di lingkungan {settings.ENVIRONMENT.upper()}! "
                "Proses dihentikan secara otomatis untuk mencegah kehilangan data produksi. "
                "Jika ingin melakukan perubahan skema, lakukan migrasi manual atau ubah nama koleksi."
            )
            # Hentikan program dengan menaikkan ValueError
            raise ValueError(
                f"Dilarang keras menimpa koleksi aktif di lingkungan {settings.ENVIRONMENT}"
            )

    logger.info(
        f"Membuat koleksi baru '{collection_name}' dengan arsitektur Hybrid (Dense + Sparse/BM25)..."
    )

    await async_qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text_dense": VectorParams(
                size=3072,  # Ukuran embedding Gemini text-embedding-004
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "text_sparse": SparseVectorParams(
                modifier=Modifier.IDF  # Mengaktifkan kalkulasi BM25 otomatis di sisi Qdrant
            )
        },
    )

    # Indeks pencarian kata kunci leksikal
    await async_qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="text_child",
        field_schema=TextIndexParams(
            type=PayloadSchemaType.TEXT, tokenizer=TokenizerType.WORD, lowercase=True
        ),
    )

    # Indeks metadata untuk filter penajaman query hukum Anda
    await async_qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="status_keberlakuan",
        field_schema="keyword",
    )
    await async_qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="uu_id",
        field_schema="keyword",
    )

    logger.info(f"Koleksi '{collection_name}' sukses dikonfigurasi.")


async def run_qdrant_initialization():
    try:
        logger.info(
            f"=== MEMULAI INISIALISASI QDRANT HYBRID [{settings.ENVIRONMENT.upper()}] ==="
        )

        # Eksekusi fungsi inisialisasi menggunakan objek koneksi pasif Anda
        await init_qdrant_collections()

        logger.info("=== QDRANT HYBRID BERHASIL DIKONFIGURASI ===")
    except Exception as e:
        logger.error(f"Inisialisasi Qdrant gagal: {e}")
        raise e
    finally:
        # Menutup HTTP client Qdrant secara bersih
        logger.info("Menutup koneksi Async Qdrant Client...")
        await async_qdrant_client.close()
        logger.info("Koneksi Async Qdrant Client berhasil ditutup bersih.")


if __name__ == "__main__":
    asyncio.run(run_qdrant_initialization())
