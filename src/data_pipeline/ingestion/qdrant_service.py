import asyncio

from google.genai import types
from qdrant_client import models

from src.config import settings
from src.database.connection import async_qdrant_client
from src.logger import logger
from src.utils.embeddings import ai_client


async def prepare_and_upload_to_qdrant(prepared_points: list[dict]):
    """
    Modul asinkron menggunakan strategi Streaming Batching.
    Langsung mengirim data per batch ke Qdrant Cloud setelah embedding selesai
    untuk mencegah Idle Connection Timeout.
    """

    # 1. Tentukan ukuran batch di bawah limit 100 RPM Free Tier Google
    BATCH_SIZE = 80

    # Pecah list prepared_points menjadi sub-list kecil berukuran 80 data
    batches = [
        prepared_points[i : i + BATCH_SIZE]
        for i in range(0, len(prepared_points), BATCH_SIZE)
    ]

    logger.info(
        f"-> Total data: {len(prepared_points)}. Dibagi menjadi {len(batches)} gelombang batch."
    )

    async def process_single_point(data: dict) -> models.PointStruct:
        """Memproses satu data point menggunakan metode asinkron (aio) resmi SDK."""
        try:
            # Memanggil API secara asinkron menggunakan properti .aio sesuai struktur utils Anda
            response = await ai_client.aio.models.embed_content(
                model="models/gemini-embedding-2",
                contents=data["text_to_embed"],
                config=types.EmbedContentConfig(output_dimensionality=3072),
            )
            dense_vector = response.embeddings[0].values

            # Kembalikan objek PointStruct yang siap di-upsert
            return models.PointStruct(
                id=data["point_id"],
                vector={
                    "text_dense": dense_vector,
                    "text_sparse": models.Document(
                        text=data["text_to_embed"], model="Qdrant/bm25"
                    ),
                },
                payload={
                    "text_child": data["text_child_raw"],
                    "text_parent": data["text_parent_utuh"],
                    "uu_id": data["uu_id"],
                    "status_keberlakuan": "berlaku",
                    "parent_id": data["parent_id"],
                    "bab": data["bab"],
                    "pasal": data["pasal"],
                },
            )
        except Exception as e:
            logger.error(
                f"[ERROR] Gagal memproses embedding untuk point {data['point_id']}: {e}"
            )
            raise e

    # 2. EKSEKUSI BATCH DEMI BATCH DAN LANGSUNG ALIRKAN KE QDRANT
    for index, batch in enumerate(batches):
        logger.info(
            f"-> Mengeksekusi gelombang batch ke-{index + 1}/{len(batches)} (Berisi {len(batch)} data)..."
        )

        # Jalankan pembentukan embedding secara paralel menggunakan asyncio.gather untuk batch ini
        tasks = [process_single_point(p) for p in batch]
        qdrant_points_batch = await asyncio.gather(*tasks)

        # KUNCI PERBAIKAN TIMEOUT: Langsung kirim batch ini ke Qdrant Cloud
        logger.info(
            f"   -> Mengunggah {len(qdrant_points_batch)} points dari batch ke-{index + 1} ke Qdrant Cloud..."
        )

        await async_qdrant_client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME, points=qdrant_points_batch
        )
        logger.info(f"   [SUKSES] Data batch ke-{index + 1} aman disimpan di Qdrant.")

        # Jika masih ada batch berikutnya, tidur selama 65 detik untuk mereset kuota RPM Google
        if index < len(batches) - 1:
            logger.info(
                "-> Jeda 65 detik untuk mereset kuota Free Tier Gemini sebelum batch berikutnya...\n"
            )
            await asyncio.sleep(65)

    logger.info(
        f"\n[SELESAI] Seluruh {len(batches)} batch berhasil ditransfer ke Qdrant Cloud!"
    )
