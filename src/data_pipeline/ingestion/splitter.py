"""Split legal documents into hierarchical child chunks for Qdrant ingestion."""

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_laws_into_hierarchical_points(
    parsed_laws: list[dict], chunk_size: int = 400, chunk_overlap: int = 80
) -> list[dict]:
    """Split parent articles into child chunks with hierarchy: UU -> BAB -> Pasal."""
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    qdrant_prepared_data = []

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

            qdrant_prepared_data.append(
                {
                    "point_id": str(uuid.uuid4()),
                    "parent_id": parent_id,
                    "text_to_embed": text_to_embed,
                    "text_child_raw": chunk,
                    "text_parent_utuh": full_parent_text,
                    "uu_id": law["uu_id"],
                    "bab": bab,
                    "pasal": pasal,
                }
            )

    return qdrant_prepared_data
