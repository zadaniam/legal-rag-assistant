# tests/unit/test_splitter.py
import pytest
from ingestion.splitter import split_laws_into_hierarchical_points

def test_split_laws_into_hierarchical_points_clean_hierarchy():
    # 1. SETUP: Mock input data pasal tanpa elemen 'bagian'
    mock_parsed_laws = [
        {
            "uu_id": "uu_bpjs_2011",
            "bab": "BAB I KETENTUAN UMUM",
            "pasal": "Pasal 1",
            "text_parent": "### Pasal 1\nDalam Undang-Undang ini yang dimaksud dengan: 1. Badan Penyelenggara Jaminan Sosial..."
        }
    ]

    # 2. ACTION: Jalankan splitter
    result_points = split_laws_into_hierarchical_points(
        parsed_laws=mock_parsed_laws, 
        chunk_size=100, 
        chunk_overlap=20
    )

    # 3. ASSERT: Validasi data output
    assert isinstance(result_points, list)
    assert len(result_points) > 0

    sample_point = result_points[0]
    
    # Pastikan struktur kunci payload yang kita kirim ke Qdrant sesuai rencana bersih kita
    expected_keys = {
        "point_id", "parent_id", "text_to_embed", "text_child_raw", 
        "text_parent_utuh", "uu_id", "bab", "pasal"
    }
    assert expected_keys.issubset(sample_point.keys())
    # Pastikan kunci 'bagian' yang sudah usang benar-benar tidak ada di data akhir
    assert "bagian" not in sample_point

    # Validasi keandalan Teks Jangkar Baru
    expected_prefix = "UU BPJS [BAB I KETENTUAN UMUM - Pasal 1]:"
    assert sample_point["text_to_embed"].startswith(expected_prefix), "Format prefix hirarki UU->BAB->Pasal salah"
