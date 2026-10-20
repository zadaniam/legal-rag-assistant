# tests/unit/test_parser.py
import os
import pytest
from ingestion.parser_rag import parse_markdown_to_object

@pytest.fixture
def sample_markdown_file(tmp_path):
    """Fixture untuk membuat file Markdown hukum tiruan untuk keperluan testing"""
    content = (
        "# UNDANG-UNDANG REPUBLIK INDONESIA NOMOR 24 TAHUN 2011\n\n"
        "## BAB I KETENTUAN UMUM\n\n"
        "### Pasal 1\n"
        "Dalam Undang-Undang ini yang dimaksud dengan: 1. BPJS adalah badan hukum...\n\n"
        "### Pasal 2\n"
        "BPJS menyelenggarakan jaminan sosial berdasarkan asas kemanusiaan.\n\n"
        "## BAB II LANDASAN\n\n"
        "### Pasal 3\n"
        "BPJS bertujuan untuk mewujudkan terselenggaranya program..."
    )
    # Membuat file temporer yang aman dan otomatis dihapus setelah test selesai
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "uu_mock_test.md"
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_parse_markdown_to_object_success(sample_markdown_file):
    # 1. SETUP
    target_uu_id = "uu_bpjs_2011"

    # 2. ACTION: Jalankan fungsi parser hukum
    parsed_result = parse_markdown_to_object(sample_markdown_file, uu_id=target_uu_id)

    # 3. ASSERT: Validasi output
    assert isinstance(parsed_result, list), "Output parser harus berbentuk List"
    assert len(parsed_result) == 3, "Harusnya menghasilkan 3 objek pasal (Pasal 1, Pasal 2, Pasal 3)"

    # --- Pengujian Pasal 1 (BAB I) ---
    pasal_1 = parsed_result[0]
    assert pasal_1["uu_id"] == target_uu_id
    assert pasal_1["judul_uu"] == "UNDANG-UNDANG REPUBLIK INDONESIA NOMOR 24 TAHUN 2011"
    assert pasal_1["bab"] == "BAB I KETENTUAN UMUM"
    assert pasal_1["pasal"] == "Pasal 1"
    assert "Dalam Undang-Undang ini yang dimaksud dengan" in pasal_1["text_parent"]

    # --- Pengujian Pasal 2 (Pastikan mewarisi BAB I yang sama) ---
    pasal_2 = parsed_result[1]
    assert pasal_2["bab"] == "BAB I KETENTUAN UMUM", "Gagal mewariskan BAB I ke pasal di bawahnya"
    assert pasal_2["pasal"] == "Pasal 2"

    # --- Pengujian Pasal 3 (Pastikan beralih ke BAB II secara dinamis) ---
    pasal_3 = parsed_result[2]
    assert pasal_3["bab"] == "BAB II LANDASAN", "Gagal beralih ke BAB II"
    assert pasal_3["pasal"] == "Pasal 3"
    assert "BPJS bertujuan untuk mewujudkan" in pasal_3["text_parent"]


def test_parse_markdown_to_object_file_not_found():
    # Menguji apakah sistem melempar error yang benar jika berkas .md tidak ada
    with pytest.raises(FileNotFoundError):
        parse_markdown_to_object("jalur/palsu/yang/tidak/eksis.md", uu_id="uu_test")
