"""Legal search node for the legal RAG agent.

Performs parallel hybrid search against Qdrant (dense semantic + sparse BM25
with RRF fusion) using queries decomposed by the context manager.
"""

import asyncio
from typing import Any

from qdrant_client import models

from src.agent.state import LegalAgentState
from src.config import settings
from src.database.connection import async_qdrant_client
from src.logger import logger
from src.utils.embeddings import embed_text


async def legal_search_node(state: LegalAgentState) -> dict:
    """Execute async Qdrant hybrid search using decomposed queries."""
    search_payloads = state.get("search_payloads", [])
    if not search_payloads:
        logger.info("No search payloads. Skipping Qdrant process.")
        return {"retrieved_documents": []}

    collection_name = settings.QDRANT_COLLECTION_NAME

    queries_to_embed = [p.get("query", "") for p in search_payloads]

    try:
        all_query_vectors = await embed_text(queries_to_embed)
    except Exception as e:
        logger.error(f"Failed to perform batch embedding in search node: {e}")
        return {"retrieved_documents": []}

    async def search_qdrant_worker(
        idx: int, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        target_uu = payload.get("target_uu", "Semua")
        query_text = payload.get("query", "")

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
                "Executing Qdrant hybrid search in parallel: "
                f"'{query_text}' | Filter: {target_uu}"
            )

            response = await async_qdrant_client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using="text_dense",  # Matches the collection initialization
                        filter=search_filter,
                        limit=20,
                    ),
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

            detailed_points = []
            for point in response.points:
                if point.payload:
                    detailed_points.append(
                        {
                            "id": point.id,
                            "score": point.score,
                            "triggered_by_query": query_text,
                            "target_uu_filter": target_uu,
                            "payload": point.payload,
                        }
                    )
            return detailed_points

        except Exception as e:
            logger.error(f"Qdrant Hybrid worker failed on query '{query_text}': {e}")
            return []

    tasks = [search_qdrant_worker(i, p) for i, p in enumerate(search_payloads)]
    batch_results = await asyncio.gather(*tasks)

    all_results = []
    seen_parents = set()

    for res_list in batch_results:
        for doc in res_list:
            payload_data = doc.get("payload", {})
            parent_id = payload_data.get("parent_id")
            point_id = doc.get("id")  # Kept for log monitoring

            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                all_results.append(payload_data)

                logger.info(
                    "-> [RETRIEVED UNIQ] Parent ID: "
                    f"{parent_id} | Point ID: {point_id} | "
                    f"Score: {doc['score']:.4f} | "
                    f"Pasal: {payload_data.get('pasal', 'N/A')} | "
                    f"Query: '{doc['triggered_by_query']}'"
                )
            else:
                logger.debug(
                    f"-> [SKIPPED DUPLICATE PASAL] Parent ID: {parent_id} "
                    "already represented."
                )

    logger.info(
        "Qdrant search completed. Collected "
        f"{len(all_results)} official legal documents without duplicate articles."
    )
    return {"retrieved_documents": all_results}
