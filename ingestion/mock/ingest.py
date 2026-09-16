# src/database/ingest.py
import uuid

from qdrant_client.models import PointStruct

from ingestion.mock.text_processor import parse_hierarchical_legal_markdown
from src.database.connection import async_qdrant_client
from src.logger import logger
from src.utils.embeddings import embed_text

COLLECTION_NAME = "legal_knowledge_base"
BATCH_SIZE = 100  # Mengunggah ke Qdrant per 100 dokumen agar efisien


async def ingest_raw_legal_text(raw_text: str, uu_id: str, nama_regulasi: str):
    chunks = parse_hierarchical_legal_markdown(raw_text, uu_id, nama_regulasi)
    if not chunks:
        logger.warning("Tidak ada chunks yang dihasilkan dari teks.")
        return

    logger.info(f"Memulai batch ingestion untuk {len(chunks)} dokumen...")

    # Proses data per-batch
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]

        # 1. Kumpulkan semua isi teks ayat (child) dari batch saat ini
        texts_to_embed = [chunk["text_child"] for chunk in batch_chunks]

        try:
            # 2. Ambil vektor secara massal dalam satu kali panggil API (Batch Embedding)
            vectors = await embed_text(texts_to_embed)

            points = []
            for idx, chunk in enumerate(batch_chunks):
                point_id = str(uuid.uuid4())

                # PERBAIKAN HYBRID SINTAKSIS TERBARU:
                # 1. Gunakan key "text_dense" sesuai nama konfigurasi di create_collection.
                # 2. Tidak perlu menyertakan key "text_sparse" manual saat upsert data.
                #    Qdrant otomatis membangun index sparse dari field "text_child" di dalam payload.
                points.append(
                    PointStruct(
                        id=point_id, vector={"text_dense": vectors[idx]}, payload=chunk
                    )
                )

            # 3. Upsert massal ke Qdrant untuk batch saat ini
            await async_qdrant_client.upsert(
                collection_name=COLLECTION_NAME, points=points
            )
            logger.info(
                f"Berhasil mengunggah batch indeks {i} sampai {i + len(batch_chunks)}"
            )

        except Exception as e:
            logger.error(f"Gagal memproses batch pada rentang indeks {i}: {e}")
            continue

    logger.info("Seluruh rangkaian Ingestion Data Selesai Sukses!")
