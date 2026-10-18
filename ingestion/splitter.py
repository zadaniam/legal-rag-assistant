# ingestion/splitter.py
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_laws_into_hierarchical_points(parsed_laws: list[dict], chunk_size: int = 400, chunk_overlap: int = 80) -> list[dict]:
    """Memotong pasal (Parent) menjadi potongan kecil (Child) dengan hirarki murni: UU -> BAB -> Pasal"""
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    qdrant_prepared_data = []

    for law in parsed_laws:
        parent_id = str(uuid.uuid4())
        teks_induk_utuh = law["text_parent"]

        # Proses pemotongan teks utama
        child_chunks = child_splitter.split_text(teks_induk_utuh)

        # Deteksi Label Nama UU secara dinamis berdasarkan uu_id
        uu_label = "UU BPJS" if "bpjs" in law["uu_id"].lower() else "UU KETENAGAKERJAAN"
        
        bab = law.get("bab", "").strip()
        pasal = law.get("pasal", "").strip()

        # Format teks jangkar super bersih: UU -> BAB -> Pasal
        prefix_konteks = f"{uu_label} [{bab} - {pasal}]: "

        for chunk in child_chunks:
            text_untuk_embedding = prefix_konteks + chunk
            
            qdrant_prepared_data.append({
                "point_id": str(uuid.uuid4()),
                "parent_id": parent_id,
                "text_to_embed": text_untuk_embedding,
                "text_child_raw": chunk,
                "text_parent_utuh": teks_induk_utuh,
                "uu_id": law["uu_id"],
                "bab": bab,
                "pasal": pasal
            })

    return qdrant_prepared_data
