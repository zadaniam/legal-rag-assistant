import os
import re


def run_cleaner(input_dir, output_file_path):
    print("Memulai proses pembersihan teks (cleaning) menggunakan Regex...")

    # Ambil semua file .txt hasil ekstraksi dan urutkan berdasarkan nama file
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")])

    if not all_files:
        print(f"Error: Tidak ada file .txt ditemukan di folder {input_dir}")
        return

    combined_text_list = []

    for file_name in all_files:
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            page_text = f.read()
            combined_text_list.append(page_text)

    # Gabungkan semua halaman menjadi satu string raksasa
    full_text = "\n".join(combined_text_list)

    # --- PROSES REGEX CLEANING ---

    # 1. Hapus nomor halaman tunggal berformat "- 14 -" atau "- 2 -" di awal/tengah baris
    full_text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", full_text, flags=re.MULTILINE)

    # 2. Hapus teks sambung gantung di pojok kanan bawah halaman (catchwords)
    # Mendeteksi titik berulang (\.{2,}) ATAU karakter ellipsis tunggal (…), diikuti spasi opsional (\s*$) di akhir baris
    full_text = re.sub(r"^.*(?:\.{2,}|…)\s*$", "", full_text, flags=re.MULTILINE)

    # 3. Memperbaiki kata yang terputus tanda hubung akibat pergantian baris (Hyphenation)
    # Contoh: "ketenaga-\nkerjaan" menjadi "ketenagakerjaan"
    full_text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", full_text)

    # 4. Merapatkan jeda enter vertikal antar-halaman
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    # 5. Menyambung baris kalimat yang terputus (Hapus enter yang tidak perlu)
    full_text = re.sub(r"(?<=[a-zA-Z,])\n(?=[a-z])", " ", full_text)

    # 6. Normalisasi spasi ganda dan tab agar teks rapi
    full_text = re.sub(r"[ \t]+", " ", full_text)

    # Simpan hasil pembersihan akhir ke dalam SATU file teks utuh
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Selesai! File teks bersih gabungan berhasil dibuat di: {output_file_path}")


if __name__ == "__main__":
    INPUT_FOLDER = "data/2_extracted/"
    OUTPUT_FILE = "data/3_cleaned/uu_ketenagakerjaan_clean.txt"

    run_cleaner(INPUT_FOLDER, OUTPUT_FILE)
