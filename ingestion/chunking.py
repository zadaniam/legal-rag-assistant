# Parent-child chunking

import json
import uuid

import torch
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# =====================================================================
# 1. LOAD DOKUMEN MARKDOWN
# =====================================================================
file_path = "uu_bpjs_siap_rag.md"
with open(file_path, "r", encoding="utf-8") as f:
    markdown_document = f.read()


# =====================================================================
# 2. PROSES PARENT CHUNKING (Berdasarkan Hierarki Hukum)
# =====================================================================
# Kita memotong berdasarkan tanda pagar untuk dijadikan Dokumen Induk (Parent)
headers_to_split_on = [
    ("#", "Bab"),
    ("##", "Bagian"),
    ("###", "Pasal"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
parent_docs = markdown_splitter.split_text(markdown_document)

print(f"Berhasil membuat {len(parent_docs)} Parent Chunks (berbasis Pasal/Bab).")


# =====================================================================
# 3. PROSES CHILD CHUNKING (Optimasi Injeksi Metadata Konteks Embedding)
# =====================================================================

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # Diperbesar agar satu poin hukum utuh
    chunk_overlap=80,  # Overlap 15-20%
    separators=["\n\n", "\n", ". ", " ", ""],
)

all_child_docs = []
parent_store = {}

for parent_doc in parent_docs:
    parent_id = str(uuid.uuid4())
    metadata_pasal = parent_doc.metadata

    parent_store[parent_id] = {
        "text": parent_doc.page_content,
        "metadata": metadata_pasal,
    }

    child_chunks = child_splitter.split_text(parent_doc.page_content)

    # Ambil informasi Bab dan Pasal untuk dijadikan teks jangkar (anchor text)
    bab_info = metadata_pasal.get("Bab", "BAB: Tidak ada")
    bagian_info = metadata_pasal.get("Bagian", "Bagian: Tidak ada")
    pasal_info = metadata_pasal.get("Pasal", "Pasal: Tidak Ada")
    prefix_konteks = f"UU BPJS [{bab_info} - {bagian_info} - {pasal_info}]: "

    for chunk in child_chunks:
        # PENTING: Gabungkan prefix konteks dengan potongan teks asli
        text_untuk_embedding = prefix_konteks + chunk

        child_metadata = metadata_pasal.copy()
        child_metadata["parent_id"] = parent_id

        # Masukkan teks yang sudah kaya konteks ke dalam dokumen child
        child_doc = Document(page_content=text_untuk_embedding, metadata=child_metadata)
        all_child_docs.append(child_doc)

print(f"Berhasil membuat {len(all_child_docs)} Child Chunks yang teroptimasi konteks.")


# =====================================================================
# 4. SIMPAN CHILD CHUNKS KE VECTOR DATABASE (ChromaDB Lokal)
# =====================================================================
# Deteksi device otomatis: mps (Mac) -> cuda (Nvidia GPU) -> cpu (fallback)
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Menggunakan device: {device}")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

vector_db = Chroma.from_documents(
    documents=all_child_docs,
    embedding=embedding_model,
    persist_directory="./chroma_db_bpjs",  # Database akan disimpan di folder lokal ini
)

print("Semua Child Chunks berhasil disimpan ke ChromaDB!")


# =====================================================================
# 5. SIMPAN PARENT STORE KE ROOT FOLDER
# =====================================================================
with open("parent_store.json", "w", encoding="utf-8") as f:
    json.dump(parent_store, f, indent=4, ensure_ascii=False)
print("-> Berhasil menyimpan file permanen: 'parent_store.json'")
