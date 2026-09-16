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

# PERBAIKAN: Hanya impor client Qdrant, buang fungsi lifespan PostgreSQL
from src.database.connection import async_qdrant_client
from src.logger import logger


async def init_qdrant_collections():
    collection_name = "legal_knowledge_base"
    logger.info(f"Memeriksa keberadaan koleksi '{collection_name}' di Qdrant Async...")

    collections_response = await async_qdrant_client.get_collections()
    existing_collections = [c.name for c in collections_response.collections]

    # KUNCI MIGRASI: Hapus koleksi jika ada untuk mendaftarkan struktur Hybrid secara bersih
    if collection_name in existing_collections:
        logger.warning(
            f"Koleksi lama '{collection_name}' terdeteksi. Menghapus untuk migrasi ke Hybrid Search..."
        )
        try:
            await async_qdrant_client.delete_collection(collection_name=collection_name)
            logger.info(f"Koleksi lama '{collection_name}' berhasil dihapus.")
        except Exception as e:
            logger.error(f"Gagal menghapus koleksi lama: {e}")
            raise e

    logger.info(
        f"Membuat koleksi baru '{collection_name}' dengan arsitektur Hybrid (Dense + Sparse/BM25)..."
    )

    # Bangun koleksi Hybrid dengan mendaftarkan dense dan sparse vector
    await async_qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text_dense": VectorParams(
                size=3072,  # Sesuai dengan ukuran teks model Anda (misal OpenAI text-embedding-3-large)
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "text_sparse": SparseVectorParams(
                modifier=Modifier.IDF  # Mengaktifkan kalkulasi BM25 otomatis di sisi Qdrant
            )
        },
    )

    # Buat Full-Text Match Index pada field text_child untuk mesin pencari kata kunci (BM25)
    await async_qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="text_child",
        field_schema=TextIndexParams(
            type=PayloadSchemaType.TEXT, tokenizer=TokenizerType.WORD, lowercase=True
        ),
    )

    # Indeks metadata untuk filter target undang-undang di Node Search
    await async_qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="status_keberlakuan",
        field_schema="keyword",
    )
    logger.info(f"Koleksi '{collection_name}' sukses dikonfigurasi.")


async def run_qdrant_initialization():
    try:
        logger.info("=== MEMULAI INISIALISASI QDRANT HYBRID ===")
        # PERBAIKAN: Tidak perlu memanggil open pool PostgreSQL di sini

        await init_qdrant_collections()

        logger.info("=== QDRANT HYBRID BERHASIL DIKONFIGURASI ===")
    except Exception as e:
        logger.error(f"Inisialisasi Qdrant gagal: {e}")
        raise e
    finally:
        # PERBAIKAN: Tutup client Qdrant secara mandiri dan bersih saat selesai
        logger.info("Menutup koneksi Async Qdrant Client...")
        await async_qdrant_client.close()
        logger.info("Koneksi Async Qdrant Client berhasil ditutup bersih.")


if __name__ == "__main__":
    asyncio.run(run_qdrant_initialization())
