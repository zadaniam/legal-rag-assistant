"""Parse Markdown legal documents into hierarchical objects for ingestion."""

from langchain_text_splitters import MarkdownHeaderTextSplitter


def parse_markdown_to_object(md_file_path: str, uu_id: str) -> list[dict]:
    """Parse a Markdown legal document into a list of law objects.

    Maps three hierarchy levels: document title -> BAB (chapter) -> Pasal (article).
    """
    with open(md_file_path, encoding="utf-8") as f:
        markdown_text = f.read()

    headers_to_split_on = [
        ("#", "judul_uu"),
        ("##", "bab"),
        ("###", "pasal"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=True
    )
    documents = splitter.split_text(markdown_text)

    parsed_laws = []
    for doc in documents:
        metadata = doc.metadata

        nama_pasal = metadata.get("pasal", "").strip()
        isi_konten_pasal = doc.page_content.strip()

        if nama_pasal:
            text_parent_clean = f"### {nama_pasal}\n{isi_konten_pasal}"
        else:
            text_parent_clean = isi_konten_pasal

        parsed_laws.append(
            {
                "uu_id": uu_id,
                "judul_uu": metadata.get("judul_uu", "Undang-Undang"),
                "bab": metadata.get("bab", "BAB: Tidak ada"),
                "pasal": nama_pasal if nama_pasal else "Pasal: Tidak ada",
                "text_parent": text_parent_clean,
            }
        )

    return parsed_laws
