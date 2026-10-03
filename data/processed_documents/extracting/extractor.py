import os

import pdfplumber


def run_extractor(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("Memulai ekstraksi PDF berbasis koordinat...")
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            width, height = page.width, page.height

            # Memotong 170 poin dari atas (header)
            # Memotong 80 (footer)

            bbox = (0, 170, width, height - 80)
            cropped_page = page.within_bbox(bbox)

            # Menambahkan parameter layout untuk memaksa pendeteksian spasi secara ketat
            text = cropped_page.extract_text(
                layout=False, x_tolerance=1.5, y_tolerance=3
            )

            # Jika halaman kosong setelah dipotong, lompati
            if not text:
                continue

            # Simpan hasil ekstraksi mentah per halaman
            file_name = f"page_{str(page_num).zfill(3)}.txt"
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

    print(f"Selesai! Periksa teks mentah per halaman di folder: {output_dir}")


if __name__ == "__main__":
    # Tentukan jalur file input dan output di sini
    INPUT_PDF = "data/1_raw/uu_nomor_13_tahun_2003.pdf"
    OUTPUT_FOLDER = "data/2_extracted/"

    # Jalankan proses ekstraksi
    run_extractor(INPUT_PDF, OUTPUT_FOLDER)
