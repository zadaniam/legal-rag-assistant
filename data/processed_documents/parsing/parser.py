import os
import re


def run_parser(input_file_path, output_file_path):
    print("Memulai proses parsing teks ke format Markdown tunggal...")

    if not os.path.exists(input_file_path):
        print(f"Error: File input tidak ditemukan di {input_file_path}")
        return

    with open(input_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # --- PROSES PARSING HIERARKI HUKUM ---

    # 1. Format Judul Utama Dokumen (Header 1)
    text = re.sub(
        r"^(UNDANG-UNDANG REPUBLIK INDONESIA)$", r"# \1", text, flags=re.MULTILINE
    )

    # 2. Format BAB menjadi Header 2 (##) dan gabungkan dengan nama bab di baris bawahnya
    # Regex ini mencari "BAB X" dan mengambil satu baris tepat di bawahnya untuk digabungkan
    text = re.sub(
        r"^BAB\s+([I|V|X|L]+)\s*\n\s*(.*?)$", r"## BAB \1 \2", text, flags=re.MULTILINE
    )

    # 3. Format Pasal menjadi Header 3 (###)
    # Contoh: "Pasal 1" di awal baris akan otomatis menjadi "### Pasal 1"
    text = re.sub(r"^(Pasal\s+\d+.*?)$", r"### \1", text, flags=re.MULTILINE)

    # 4. Merapikan format spasi setelah penomoran ayat atau poin definisi agar seragam
    text = re.sub(r"^(\(\d+\))\s+", r"\1 ", text, flags=re.MULTILINE)
    text = re.sub(r"^(\d+\.)\s+", r"\1 ", text, flags=re.MULTILINE)

    # Simpan hasil akhir menjadi satu file .md tunggal
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Selesai! Berhasil membuat file Markdown gabungan di: {output_file_path}")


if __name__ == "__main__":
    INPUT_FILE = "data/3_cleaned/uu_ketenagakerjaan_clean.txt"
    OUTPUT_MD = "data/4_processed/uu_ketenagakerjaan_parse.md"

    run_parser(INPUT_FILE, OUTPUT_MD)
