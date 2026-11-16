"""Split legal documents into hierarchical child chunks for Qdrant ingestion."""

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.database.vector_schema import (
    LegalChunkPayload,
    LegalStatus,
    PreparedVectorPoint,
)


def split_laws_into_hierarchical_points(
    parsed_laws: list[dict], chunk_size: int = 400, chunk_overlap: int = 80
) -> list[PreparedVectorPoint]:
    """Split parent articles into child chunks with hierarchy: UU -> BAB -> Pasal."""
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    qdrant_prepared_data: list[PreparedVectorPoint] = []

    for law in parsed_laws:
        parent_id = str(uuid.uuid4())
        full_parent_text = law["text_parent"]

        child_chunks = child_splitter.split_text(full_parent_text)

        uu_label = "UU BPJS" if "bpjs" in law["uu_id"].lower() else "UU KETENAGAKERJAAN"

        bab = law.get("bab", "").strip()
        pasal = law.get("pasal", "").strip()

        context_prefix = f"{uu_label} [{bab} - {pasal}]: "

        for chunk in child_chunks:
            text_to_embed = context_prefix + chunk

            payload = LegalChunkPayload(
                text_child=chunk,
                text_parent=full_parent_text,
                uu_id=law["uu_id"],
                status_keberlakuan=LegalStatus.BERLAKU.value,
                parent_id=parent_id,
                bab=bab,
                pasal=pasal,
            )

            qdrant_prepared_data.append(
                PreparedVectorPoint(
                    point_id=str(uuid.uuid4()),
                    text_to_embed=text_to_embed,
                    payload=payload,
                )
            )

    return qdrant_prepared_data
