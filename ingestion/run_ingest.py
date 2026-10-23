# ingestion/run_ingest.py
import asyncio
import os

from ingestion.config import settings
from ingestion.parser_rag import parse_markdown_to_object
from ingestion.qdrant_service import prepare_and_upload_to_qdrant
from ingestion.splitter import split_laws_into_hierarchical_points
from src.database.connection import async_qdrant_client


async def run_ingest():
    print(f"[START] Pipeline Ingestion aktif pada mode: {settings.ENVIRONMENT.upper()}")

    # KUNCI PERBAIKAN: Buat daftar berkas dokumen yang ingin dimasukkan ke database RAG
    FILES_TO_INGEST = [
        {"path": "data/processed/uu_bpjs_rag.md", "uu_id": "uu_bpjs_2011"},
        {
            "path": "data/processed/uu_ketenagakerjaan_rag.md",
            "uu_id": "uu_ketenagakerjaan_2003",
        },
    ]

    try:
        # Lakukan looping untuk mengeksekusi setiap dokumen satu per satu
        for doc in FILES_TO_INGEST:
            target_file = doc["path"]
            target_id = doc["uu_id"]

            if not os.path.exists(target_file):
                print(f"[SKIP] Berkas tidak ditemukan, melewati: {target_file}")
                continue

            print(
                f"\n--- Memproses Dokumen: {os.path.basename(target_file)} ({target_id}) ---"
            )

            # 1. Eksekusi Modul Parser Hirarki Anda (Membawa uu_id dinamis)
            parsed_laws = parse_markdown_to_object(target_file, uu_id=target_id)

            # 2. Eksekusi Modul Splitter Teks Jangkar Anda
            prepared_points = split_laws_into_hierarchical_points(parsed_laws)

            # 3. Eksekusi Modul Service Database secara Asinkron
            print(
                f"-> Memproses embedding Gemini & mengirim {len(prepared_points)} data ke Qdrant..."
            )
            await prepare_and_upload_to_qdrant(prepared_points)

            print(f"[SUKSES] Data hukum untuk {target_id} berhasil dimuat ke database!")

    except Exception as e:
        print(f"[ERROR] Proses ingestion massal gagal: {e}")
        raise e
    finally:
        print("\nMenutup koneksi Async Qdrant Client...")
        await async_qdrant_client.close()
        print("Koneksi Async Qdrant Client berhasil ditutup bersih.")


if __name__ == "__main__":
    asyncio.run(run_ingest())
