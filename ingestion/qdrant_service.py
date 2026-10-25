# ingestion/qdrant_service.py
import asyncio

from google import genai
from google.genai import types
from qdrant_client import models

from src.config import settings
from src.database.connection import async_qdrant_client


async def prepare_and_upload_to_qdrant(prepared_points: list[dict]):
    """Modul asinkron menggunakan strategi Time-Window Batching untuk Free Tier Google."""
    ai_client = genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())
    qdrant_points = []

    # 1. STRATEGI PILIHAN ANDA: Tentukan ukuran batch di bawah limit 100 RPM
    BATCH_SIZE = 80

    # Pecah list prepared_points menjadi sub-list kecil berukuran 80 data
    batches = [
        prepared_points[i : i + BATCH_SIZE]
        for i in range(0, len(prepared_points), BATCH_SIZE)
    ]

    print(
        f"-> Total data: {len(prepared_points)}. Dibagi menjadi {len(batches)} gelombang batch."
    )

    async def process_single_point(data: dict):
        try:
            response = ai_client.models.embed_content(
                model="models/gemini-embedding-2",
                contents=data["text_to_embed"],
                config=types.EmbedContentConfig(output_dimensionality=3072),
            )
            dense_vector = response.embeddings[0].values

            qdrant_points.append(
                models.PointStruct(
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
            )
        except Exception as e:
            print(f"[ERROR] Gagal memproses data point {data['point_id']}: {e}")
            raise e

    # 2. EKSEKUSI BATCH DEMI BATCH DENGAN JEDA WAKTU TUNGGU
    for index, batch in enumerate(batches):
        print(
            f"-> Mengeksekusi gelombang batch ke-{index + 1}/{len(batches)} (Berisi {len(batch)} data)..."
        )

        # Jalankan satu batch (maksimal 80 request) secara paralel simultan
        tasks = [process_single_point(p) for p in batch]
        await asyncio.gather(*tasks)

        # Jika masih ada batch berikutnya, tidur selama 65 detik untuk mereset kuota RPM Google
        if index < len(batches) - 1:
            print(
                f"-> Batch {index + 1} selesai. Menunggu 65 detik agar kuota Free Tier Google di-reset..."
            )
            await asyncio.sleep(65)

    # 3. KIRIM MASSAL KE QDRANT SEKALIGUS
    print(f"-> Mengunggah total {len(qdrant_points)} points matang ke Qdrant...")
    await async_qdrant_client.upsert(
        collection_name=settings.QDRANT_COLLECTION_NAME, points=qdrant_points
    )
