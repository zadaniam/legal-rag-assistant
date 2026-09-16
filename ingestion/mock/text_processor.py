# src/utils/text_processor.py

import re
from typing import Any

from src.logger import logger


def parse_hierarchical_legal_markdown(
    text: str, uu_id: str, nama_regulasi: str
) -> list[dict[str, Any]]:
    """
    Memotong dokumen hukum berbasis Markdown menjadi skema terstruktur Parent-Child.
    Parent chunk = Satu Pasal Utuh (Ayat 1 sampai selesai).
    Child chunk = Tiap Ayat secara terpisah (Vektor pencarian).
    """
    logger.info(f"Memulai pemrosesan dokumen hierarkis untuk {uu_id}...")

    chunks_payload = []

    # 1. Ekstrak BAB secara global untuk melacak posisi teks berada di BAB mana
    # Mencari pola baris yang diawali dengan "BAB " sampai akhir baris
    bab_positions = list(re.finditer(r"^BAB\s+[^\n]+", text, flags=re.MULTILINE))

    # 2. Cari semua posisi "Pasal X" secara global di dalam dokumen menggunakan finditer
    pasal_positions = list(re.finditer(r"^Pasal\s+(\d+)", text, flags=re.MULTILINE))

    if not pasal_positions:
        logger.warning(
            f"Tidak ditemukan penanda Pasal yang valid pada regulasi {nama_regulasi}"
        )
        return []

    for i, match in enumerate(pasal_positions):
        nomor_pasal = match.group(1)
        start_pos = match.start()

        # Tentukan batas akhir teks pasal (sampai ketemu pasal berikutnya atau akhir teks)
        end_pos = (
            pasal_positions[i + 1].start()
            if i + 1 < len(pasal_positions)
            else len(text)
        )
        segment = text[start_pos:end_pos].strip()
        text_parent = segment  # Pasal utuh bertindak sebagai Parent Chunk

        # 3. Cari tahu bab saat ini secara dinamis berdasarkan posisi karakter pasal
        current_bab = "Tidak Terdefinisi"
        for bab_match in bab_positions:
            if bab_match.start() < start_pos:
                current_bab = bab_match.group(0).strip()
            else:
                break

        # 4. Pecah segmen pasal menjadi ayat-ayat kecil (Child Chunks)
        # Menangkap pola penanda ayat seperti (1) isi teks, (2) isi teks
        ayat_splits = re.findall(r"\((\d+)\)\s+([^\n]+)", segment)

        if ayat_splits:
            for nomor_ayat, isi_ayat in ayat_splits:
                chunks_payload.append(
                    {
                        "uu_id": uu_id,
                        "nama_regulasi": nama_regulasi,
                        "bab": current_bab,
                        "pasal": nomor_pasal,
                        "ayat": nomor_ayat,
                        "text_child": f"Pasal {nomor_pasal} Ayat ({nomor_ayat}): {isi_ayat.strip()}",
                        "text_parent": text_parent,
                        "status_keberlakuan": "aktif",
                    }
                )
        else:
            # Jika pasal tersebut tidak memiliki sub-ayat angka (pasal tunggal langsung teks)
            # Bersihkan baris pertama judul pasal untuk mengambil isinya secara bersih
            clean_text = re.sub(r"^Pasal\s+\d+\s*", "", segment).strip()
            chunks_payload.append(
                {
                    "uu_id": uu_id,
                    "nama_regulasi": nama_regulasi,
                    "bab": current_bab,
                    "pasal": nomor_pasal,
                    "ayat": "1",  # Default sebagai ayat tunggal
                    "text_child": f"Pasal {nomor_pasal}: {clean_text}",
                    "text_parent": text_parent,
                    "status_keberlakuan": "aktif",
                }
            )

    logger.info(
        f"Selesai memproses. Berhasil memecah menjadi {len(chunks_payload)} Child-Ayat Chunks."
    )
    return chunks_payload
