import pdfplumber

file = "data/1_raw/uu_nomor_13_tahun_2003.pdf"

halaman = 1
baris_atas = 3
baris_bawah = 2

with pdfplumber.open(file) as pdf:
    page = pdf.pages[halaman - 1]

    print(f"\nLebar Halaman: {page.width} poin")
    print(f"Tinggi Halaman: {page.height} poin")

    print(f"\n--- Posisi {baris_atas} Baris Teks Teratas ---")
    # Mengambil koordinat setiap baris teks
    for obj in page.extract_text_lines()[:baris_atas]:
        print(f"Teks: '{obj['text']}' -> {obj['top']:.2f} poin dari atas")

    print(f"\n--- Posisi {baris_bawah} Baris Teks Terbawah ---")
    # Mengambil koordinat setiap baris teks
    for obj in page.extract_text_lines()[-baris_bawah:]:
        print(f"Teks: '{obj['text']}' -> {page.height - obj['bottom']} poin dari bawah")
