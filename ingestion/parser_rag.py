# ingestion/parser_rag.py
from langchain_text_splitters import MarkdownHeaderTextSplitter


def parse_markdown_to_object(md_file_path: str, uu_id: str) -> list[dict]:
    with open(md_file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # Murni memetakan 3 tingkatan utama: Judul Utama -> Bab -> Pasal
    headers_to_split_on = [
        ("#", "judul_uu"),
        ("##", "bab"),
        ("###", "pasal"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    documents = splitter.split_text(markdown_text)

    parsed_laws = []
    for doc in documents:
        # LangChain otomatis mewariskan metadata Bab ke dalam Pasal di bawahnya
        metadata = doc.metadata

        parsed_laws.append(
            {
                "uu_id": uu_id,
                "judul_uu": metadata.get("judul_uu", "Undang-Undang"),
                "bab": metadata.get("bab", "BAB: Tidak ada"),
                "pasal": metadata.get("pasal", "Pasal: Tidak ada"),
                "text_parent": doc.page_content.strip(),  # Menghasilkan teks satu pasal utuh (Parent Chunk)
            }
        )

    return parsed_laws
