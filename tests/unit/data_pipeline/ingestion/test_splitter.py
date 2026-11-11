from src.data_pipeline.ingestion.splitter import split_laws_into_hierarchical_points


def test_split_laws_into_hierarchical_points_clean_hierarchy():
    mock_parsed_laws = [
        {
            "uu_id": "uu_bpjs_2011",
            "bab": "BAB I KETENTUAN UMUM",
            "pasal": "Pasal 1",
            "text_parent": (
                "### Pasal 1\nDalam Undang-Undang ini yang dimaksud dengan: 1. "
                "Badan Penyelenggara Jaminan Sosial..."
            ),
        }
    ]

    result_points = split_laws_into_hierarchical_points(
        parsed_laws=mock_parsed_laws, chunk_size=100, chunk_overlap=20
    )

    assert isinstance(result_points, list)
    assert len(result_points) > 0

    sample_point = result_points[0]

    expected_keys = {
        "point_id",
        "parent_id",
        "text_to_embed",
        "text_child_raw",
        "text_parent_utuh",
        "uu_id",
        "bab",
        "pasal",
    }
    assert expected_keys.issubset(sample_point.keys())
    assert "bagian" not in sample_point

    expected_prefix = "UU BPJS [BAB I KETENTUAN UMUM - Pasal 1]:"
    assert sample_point["text_to_embed"].startswith(expected_prefix), (
        "Format prefix hirarki UU->BAB->Pasal salah"
    )
