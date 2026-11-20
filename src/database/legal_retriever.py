"""Repository abstraction and Qdrant implementation for legal document retrieval."""

import asyncio
from typing import Any, Protocol

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import Condition

from src.config import settings
from src.database.connection import async_qdrant_client
from src.database.vector_schema import (
    EmbeddingSpec,
    LegalChunkPayload,
    LegalStatus,
    PayloadField,
    VectorName,
)
from src.logger import logger
from src.utils.embeddings import embed_text


class LegalRetriever(Protocol):
    """Protocol defining the interface for legal document retrieval backends."""

    async def hybrid_search(
        self,
        search_payloads: list[dict[str, Any]],
        prefetch_limit: int,
        fusion_limit: int,
    ) -> list[dict[str, Any]]:
        """Retrieve unique legal documents matching decomposed search queries."""
        ...


class QdrantLegalRetriever:
    """Async Qdrant implementation performing parallel dense + sparse RRF search."""

    def __init__(self, client: AsyncQdrantClient, collection_name: str) -> None:
        """Initialize the Qdrant retriever with an active client and collection."""
        self._client = client
        self._collection_name = collection_name

    async def hybrid_search(
        self,
        search_payloads: list[dict[str, Any]],
        prefetch_limit: int,
        fusion_limit: int,
    ) -> list[dict[str, Any]]:
        """Execute parallel hybrid search and deduplicate results by parent article."""
        if not search_payloads:
            return []

        queries_to_embed = [p.get("query", "") for p in search_payloads]

        try:
            all_query_vectors = await embed_text(queries_to_embed)
        except Exception as e:
            logger.error(f"Failed to perform batch embedding in retriever: {e}")
            return []

        tasks = [
            self._search_single_payload(
                payload=payload,
                query_vector=all_query_vectors[idx],
                prefetch_limit=prefetch_limit,
                fusion_limit=fusion_limit,
            )
            for idx, payload in enumerate(search_payloads)
        ]
        batch_results = await asyncio.gather(*tasks)

        return self._deduplicate_by_parent(batch_results)

    async def _search_single_payload(
        self,
        payload: dict[str, Any],
        query_vector: list[float],
        prefetch_limit: int,
        fusion_limit: int,
    ) -> list[dict[str, Any]]:
        """Query Qdrant for a single decomposed search payload using RRF fusion."""
        target_uu = payload.get("target_uu", LegalStatus.SEMUA.value)
        query_text = payload.get("query", "")

        must_conditions: list[Condition] = [
            models.FieldCondition(
                key=PayloadField.STATUS_KEBERLAKUAN.value,
                match=models.MatchValue(value=LegalStatus.BERLAKU.value),
            )
        ]

        if target_uu != LegalStatus.SEMUA.value:
            must_conditions.append(
                models.FieldCondition(
                    key=PayloadField.UU_ID.value,
                    match=models.MatchValue(value=target_uu),
                )
            )

        search_filter = models.Filter(must=must_conditions)

        try:
            logger.info(
                "Executing Qdrant hybrid search in parallel: "
                f"'{query_text}' | Filter: {target_uu}"
            )

            response = await self._client.query_points(
                collection_name=self._collection_name,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using=VectorName.DENSE.value,
                        filter=search_filter,
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=models.Document(
                            text=query_text, model=EmbeddingSpec.SPARSE_MODEL
                        ),
                        using=VectorName.SPARSE.value,
                        filter=search_filter,
                        limit=prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=fusion_limit,
                with_payload=True,
            )

            detailed_points: list[dict[str, Any]] = []
            for point in response.points:
                if point.payload:
                    validated_payload = LegalChunkPayload.model_validate(point.payload)
                    detailed_points.append(
                        {
                            "id": point.id,
                            "score": point.score,
                            "triggered_by_query": query_text,
                            "target_uu_filter": target_uu,
                            "payload": validated_payload.to_qdrant_payload(),
                        }
                    )
            return detailed_points

        except Exception as e:
            logger.error(f"Qdrant Hybrid worker failed on query '{query_text}': {e}")
            return []

    @staticmethod
    def _deduplicate_by_parent(
        batch_results: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Merge batch query results while preserving unique parent articles."""
        all_results: list[dict[str, Any]] = []
        seen_parents: set[str] = set()

        for res_list in batch_results:
            for doc in res_list:
                payload_data: dict[str, Any] = doc.get("payload", {})
                parent_id = payload_data.get(PayloadField.PARENT_ID.value)
                point_id = doc.get("id")

                if parent_id and parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    all_results.append(payload_data)

                    logger.info(
                        "-> [RETRIEVED UNIQ] Parent ID: "
                        f"{parent_id} | Point ID: {point_id} | "
                        f"Score: {doc['score']:.4f} | "
                        f"Pasal: {payload_data.get(PayloadField.PASAL.value, 'N/A')} | "
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
        return all_results


def get_default_legal_retriever() -> QdrantLegalRetriever:
    """Construct the default Qdrant retriever using the active global client."""
    if async_qdrant_client is None:
        raise RuntimeError(
            "Database connection error: 'async_qdrant_client' is not initialized. "
            "Ensure database pool is active before running agent search node."
        )
    return QdrantLegalRetriever(
        client=async_qdrant_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
    )
