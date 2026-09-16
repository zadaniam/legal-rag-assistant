# src/database/run_mock_ingest.py
import asyncio
import os
import sys

# Jalur dinamis agar Python mengenali folder root sebagai sistem modul utama
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Nonaktifkan tracing LangSmith sementara agar proses ingest data tidak mengotori dashboard log chat Anda
os.environ["LANGSMITH_TRACING"] = "false"

from ingestion.mock.ingest import ingest_raw_legal_text
from ingestion.mock.mock_data import MOCK_REGULASI_13_2003, MOCK_REGULASI_35_2021

# PERBAIKAN: Hanya impor objek client Qdrant langsung, buang fungsi lifespan PostgreSQL
from src.database.connection import async_qdrant_client
from src.logger import logger


async def main():
    logger.info("=== BACKEND MOCK INGESTION SYSTEM INITIALIZATION ===")

    try:
        # PERBAIKAN: Tidak perlu membuka pool koneksi PostgreSQL yang sedang mati

        # Menggunakan ID Statis sesuai instruksi Anda
        uu_id_13_2003 = "UU Ketenagakerjaan"
        uu_id_35_2021 = "UU Cipta Kerja"

        logger.info(f"Menggunakan ID UU: {uu_id_13_2003}")
        logger.info(f"Menggunakan ID UU: {uu_id_35_2021}")

        # 2. Proses Ingest data UU Ketenagakerjaan (Cuti Hamil)
        logger.info("Memproses Ingest data UU No 13 Tahun 2003...")
        await ingest_raw_legal_text(
            raw_text=MOCK_REGULASI_13_2003,
            uu_id=uu_id_13_2003,
            nama_regulasi="UU No 13 Tahun 2003",
        )

        # 3. Proses Ingest data PP Pesangon / Turunan Cipta Kerja
        logger.info("Memproses Ingest data PP No 35 Tahun 2021...")
        await ingest_raw_legal_text(
            raw_text=MOCK_REGULASI_35_2021,
            uu_id=uu_id_35_2021,
            nama_regulasi="PP No 35 Tahun 2021",
        )

        logger.info("=== SEMUA PROSES MOCK INGESTION BERHASIL ===")

    except Exception as e:
        logger.error(f"Terjadi kegagalan sistem saat eksekusi uji coba database: {e}")
    finally:
        # PERBAIKAN: Tutup client Qdrant secara mandiri agar terminal tidak menggantung
        logger.info("Menutup koneksi Async Qdrant Client...")
        await async_qdrant_client.close()
        logger.info("Koneksi Async Qdrant Client berhasil ditutup bersih.")


if __name__ == "__main__":
    asyncio.run(main())
