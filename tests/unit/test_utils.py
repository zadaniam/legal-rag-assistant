# tests/test_utils.py
from ingestion.mock.text_processor import parse_hierarchical_legal_markdown


def test_parse_hierarchical_legal_markdown_success():
    # 1. Simulasikan isi draf dokumen Markdown hukum palsu dengan struktur ketat
    mock_markdown_uu = """BAB IV
KETENAGAKERJAAN

Pasal 156
(1) Dalam hal terjadi PHK, pengusaha wajib membayar pesangon.
(2) Uang pesangon dimaksud sebesar 1 bulan upah.

Pasal 157
Upah yang digunakan sebagai dasar perhitungan pesangon terdiri atas upah pokok.
"""

    # 2. Jalankan fungsi parser utilitas
    results = parse_hierarchical_legal_markdown(
        text=mock_markdown_uu,
        uu_id="UU_TEST_2026",
        nama_regulasi="UU Uji Coba Cipta Kerja",
    )

    # 3. Validasi keakuratan pemecahan ayat (Harus menghasilkan total 3 ayat/child chunks)
    assert len(results) == 3

    # Cek pasal 156 ayat 1
    # UBAH BARIS INI: Sesuaikan ekspektasi menjadi "BAB IV" murni
    assert results[0]["bab"] == "BAB IV"
    assert results[0]["pasal"] == "156"
    assert results[0]["ayat"] == "1"
    assert "Pasal 156 Ayat (1):" in results[0]["text_child"]

    # Memastikan data pasal utuh tersimpan lengkap di text_parent
    assert "Uang pesangon dimaksud sebesar 1 bulan upah." in results[0]["text_parent"]

    # Cek pasal tunggal tanpa ayat angka (Pasal 157)
    assert results[2]["pasal"] == "157"
    assert results[2]["ayat"] == "1"
    assert "Upah yang digunakan sebagai dasar" in results[2]["text_child"]
