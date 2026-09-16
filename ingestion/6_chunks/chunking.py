import json
import os
import re

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def run_parser_to_chunks(md_file_path, output_json_path, nama_uu):
    print(
        f"[*] Memulai proses Markdown Chunking + Sub-splitting lokal untuk: {nama_uu}..."
    )

    if not os.path.exists(md_file_path):
        print(f"[ERROR] Berkas Markdown {md_file_path} tidak ditemukan.")
        return 0

    with open(md_file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # 1. SPLITTER TAHAP 1: Pemotongan Utama Berdasarkan Struktur Markdown
    headers_to_split_on = [
        ("#", "judul_dokumen"),
        ("##", "bab"),
        ("###", "pasal_lengkap"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    langchain_docs = markdown_splitter.split_text(markdown_text)

    # 2. SPLITTER TAHAP 2: Batasi Maksimal Karakter per Chunk (Sub-splitting)
    # Kita batasi idealnya maksimal 1000 - 1200 karakter dengan sedikit overlap agar konteks menyambung
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Maksimal target panjang karakter per anak chunk
        chunk_overlap=200,  # Karakter yang beririsan agar informasi di ujung tidak putus makna
        separators=["\n\n", "\n", " ", ""],
    )

    records_to_store = []

    # 3. PROSES STRUKTURISASI DAN EKSTRAKSI METADATA
    for i, doc in enumerate(langchain_docs):
        pasal_raw = doc.metadata.get("pasal_lengkap", "Pembuka/Lainnya")
        match_angka_pasal = re.search(r"\d+", pasal_raw)
        nomor_pasal = match_angka_pasal.group(0) if match_angka_pasal else pasal_raw

        # Jalankan sub-splitter pada teks pasal ini
        sub_chunks = sub_splitter.split_text(doc.page_content)

        for sub_idx, sub_text in enumerate(sub_chunks):
            # Jika pasal pendek dan tidak terpecah, ID tetap normal.
            # Jika pasal panjang dan terpecah, berikan suffix penanda sub-chunk (misal: chunk_5_0, chunk_5_1)
            if len(sub_chunks) == 1:
                record_id = f"{nama_uu}_chunk_{i}"
            else:
                record_id = f"{nama_uu}_chunk_{i}_{sub_idx}"

            # MEMBANGUN FLAT RECORD (Sesuai spesifikasi format Pinecone Integrated Inference v7.x)
            # Setiap anak chunk yang terpecah akan tetap membawa warisan metadata Bab dan Pasal induknya!
            records_to_store.append(
                {
                    "id": record_id,
                    "text": sub_text,
                    "source_doc": nama_uu,
                    "judul_dokumen": doc.metadata.get(
                        "judul_dokumen", "UU Ketenagakerjaan"
                    ),
                    "bab": doc.metadata.get("bab", "Tanpa Bab"),
                    "pasal": nomor_pasal,
                }
            )

    # 4. SIMPAN KE FILE JSON LOKAL
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(records_to_store, f, indent=4, ensure_ascii=False)

    print(
        "[SUCCESS] Berhasil memotong dokumen dengan metode Hybrid Markdown Sub-Splitting!"
    )
    print(
        f"[+] Total chunk setelah dibatasi karakter: {len(records_to_store)} potongan."
    )
    return len(records_to_store)


if __name__ == "__main__":
    INPUT_MD = "data/5_preprocessed/uu_ketenagakerjaan_preprocessed.md"
    OUTPUT_JSON = "data/6_chunks/uu_ketenagakerjaan_records.json"
    NAMA_DOKUMEN = "uu_ketenagakerjaan_13_2003"

    run_parser_to_chunks(INPUT_MD, OUTPUT_JSON, NAMA_DOKUMEN)
