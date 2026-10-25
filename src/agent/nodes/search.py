# src/agent/nodes/search.py
import asyncio
from typing import Any

from qdrant_client import models

from src.agent.state import LegalAgentState
from src.config import settings
from src.database.connection import async_qdrant_client
from src.logger import logger
from src.utils.embeddings import embed_text


async def legal_search_node(state: LegalAgentState) -> dict:
    logger.info(
        "=== Node 3: Mengeksekusi Async Qdrant Hybrid Search (Paralel Terpadu) ==="
    )

    search_payloads = state.get("search_payloads", [])
    if not search_payloads:
        logger.info("Tidak ada payload pencarian. Melewati proses Qdrant.")
        return {"retrieved_documents": []}

    collection_name = settings.QDRANT_COLLECTION_NAME 

    # 1. Kumpulkan semua string query dari seluruh payload hasil decomposition
    queries_to_embed = [p.get("query", "") for p in search_payloads]

    try:
        # 2. Ambil SEMUA vektor sekaligus secara paralel murni (Output: [[v1], [v2], [v3], ...])
        all_query_vectors = await embed_text(queries_to_embed)
    except Exception as e:
        logger.error(f"Gagal melakukan batch embedding di Node Search: {e}")
        return {"retrieved_documents": []}

    # Worker function untuk menembak Qdrant berdasarkan indeks vektor dan payload filter masing-masing
    async def search_qdrant_worker(
        idx: int, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        target_uu = payload.get("target_uu", "Semua")
        query_text = payload.get("query", "")

        # Ambil Regular Vector 1D murni [...] yang pas pasangannya berdasarkan indeks
        query_vector = all_query_vectors[idx]

        must_conditions = [
            models.FieldCondition(
                key="status_keberlakuan", match=models.MatchValue(value="berlaku")
            )
        ]
        if target_uu != "Semua":
            must_conditions.append(
                models.FieldCondition(
                    key="uu_id", match=models.MatchValue(value=target_uu)
                )
            )

        search_filter = models.Filter(must=must_conditions)

        try:
            logger.info(
                f"Menembak Qdrant Hybrid Search secara paralel: '{query_text}' | Filter: {target_uu}"
            )

            response = await async_qdrant_client.query_points(
                collection_name=collection_name,
                prefetch=[
                    # Prefetch 1: Dense Semantic Search
                    models.Prefetch(
                        query=query_vector,
                        using="text_dense",  # Disesuaikan dengan inisialisasi koleksi
                        filter=search_filter,
                        limit=20,
                    ),
                    # Prefetch 2: Sparse Keyword Search (BM25 Server-Side)
                    models.Prefetch(
                        query=models.Document(text=query_text, model="Qdrant/bm25"),
                        using="text_sparse",
                        filter=search_filter,
                        limit=20,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=3,
                with_payload=True,
            )

            # 🎯 PERBAIKAN: Menyusun detail informasi dokumen hasil retrieval (Hanya untuk keperluan LOGGING lokal)
            detailed_points = []
            for point in response.points:
                if point.payload:
                    detailed_points.append(
                        {
                            "id": point.id,
                            "score": point.score,  # Skor hasil penggabungan RRF
                            "triggered_by_query": query_text,  # Query pencari asal dari dekomposisi
                            "target_uu_filter": target_uu,
                            "payload": point.payload,  # Isi pasal hukum asli
                        }
                    )
            return detailed_points

        except Exception as e:
            logger.error(f"Worker Qdrant Hybrid gagal pada kueri '{query_text}': {e}")
            return []

    # 3. Eksekusi seluruh pencarian ke Qdrant secara bersamaan (Simultan)
    tasks = [search_qdrant_worker(i, p) for i, p in enumerate(search_payloads)]
    batch_results = await asyncio.gather(*tasks)

    # 4. Satukan dokumen hasil pencarian dan eliminasi duplikasi data berdasarkan konteks induk (Parent)
    all_results = []
    seen_parents = set()  

    for res_list in batch_results:
        for doc in res_list:
            # Ambil parent_id dari dalam payload dokumen
            payload_data = doc.get("payload", {})
            parent_id = payload_data.get("parent_id")
            point_id = doc.get("id") # Tetap diambil untuk log monitoring

            # JIKA parent_id belum pernah diambil, loloskan dokumen
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                all_results.append(payload_data)

                logger.info(
                    f"-> [RETRIEVED UNIQ] Parent ID: {parent_id} | Point ID: {point_id} | "
                    f"Score: {doc['score']:.4f} | "
                    f"Pasal: {payload_data.get('pasal', 'N/A')} | "
                    f"Query: '{doc['triggered_by_query']}'"
                )
            else:
                logger.debug(f"-> [SKIPPED DUPLICATE PASAL] Parent ID: {parent_id} sudah terwakili.")

    logger.info(
        f"Qdrant Search selesai. Berhasil mengumpulkan {len(all_results)} dokumen hukum resmi tanpa duplikasi pasal."
    )
    return {"retrieved_documents": all_results}
