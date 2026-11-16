from qdrant_client import models

from src.database.vector_schema import (
    EmbeddingSpec,
    LegalChunkPayload,
    LegalStatus,
    PayloadField,
    PreparedVectorPoint,
    QdrantCollectionSchema,
    VectorName,
)


def test_vector_collection_schema_definitions():
    vectors_cfg = QdrantCollectionSchema.vectors_config()
    assert VectorName.DENSE.value in vectors_cfg
    dense_params = vectors_cfg[VectorName.DENSE.value]
    assert dense_params.size == EmbeddingSpec.DENSE_DIMENSION == 3072
    assert dense_params.distance == models.Distance.COSINE

    sparse_cfg = QdrantCollectionSchema.sparse_vectors_config()
    assert VectorName.SPARSE.value in sparse_cfg
    assert sparse_cfg[VectorName.SPARSE.value].modifier == models.Modifier.IDF

    indexes = QdrantCollectionSchema.payload_indexes()
    indexed_fields = [field_name for field_name, _ in indexes]
    assert indexed_fields == [
        PayloadField.TEXT_CHILD.value,
        PayloadField.STATUS_KEBERLAKUAN.value,
        PayloadField.UU_ID.value,
    ]


def test_legal_chunk_payload_backward_compatibility():
    # Simulates legacy/partial payload retrieved from an existing Qdrant collection
    legacy_raw_payload = {
        "text_child": "Isi ayat lama",
        "text_parent": "Pasal 10\nIsi pasal lengkap",
        "uu_id": "uu_bpjs_2011",
        "parent_id": "legacy-parent-123",
        "legacy_extra_field": "nilai_lama_yang_masih_ada",
    }

    validated = LegalChunkPayload.model_validate(legacy_raw_payload)
    serialized = validated.to_qdrant_payload()

    assert serialized["text_child"] == "Isi ayat lama"
    assert serialized["status_keberlakuan"] == LegalStatus.BERLAKU.value
    assert serialized["legacy_extra_field"] == "nilai_lama_yang_masih_ada"


def test_prepared_vector_point_serialization():
    point = PreparedVectorPoint(
        point_id="pt-001",
        text_to_embed="UU BPJS [BAB I - Pasal 1]: Teks",
        payload=LegalChunkPayload(
            text_child="Teks",
            text_parent="Pasal 1\nTeks",
            uu_id="uu_bpjs_2011",
            parent_id="parent-001",
            bab="BAB I",
            pasal="Pasal 1",
        ),
    )

    dumped = point.payload.to_qdrant_payload()
    assert set(dumped.keys()) == {
        "text_child",
        "text_parent",
        "uu_id",
        "status_keberlakuan",
        "parent_id",
        "bab",
        "pasal",
    }
