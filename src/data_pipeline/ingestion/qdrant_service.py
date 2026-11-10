"""Asynchronous Qdrant upload module using streaming batch strategy.

Sends data per batch to Qdrant Cloud after embedding completes to prevent
Idle Connection Timeout.
"""

import asyncio

from google.genai import types
from qdrant_client import models

from src.config import settings
from src.database.connection import async_qdrant_client
from src.logger import logger
from src.utils.embeddings import ai_client


async def prepare_and_upload_to_qdrant(prepared_points: list[dict]) -> None:
    """Upload prepared points to Qdrant using streaming batch strategy.

    Embeds and uploads data in batches to avoid Idle Connection Timeout
    and respect Google Free Tier RPM limits.
    """
    batch_size = 80

    batches = [
        prepared_points[i : i + batch_size]
        for i in range(0, len(prepared_points), batch_size)
    ]

    logger.info(
        f"-> Total data: {len(prepared_points)}. Split into {len(batches)} batch waves."
    )

    async def process_single_point(data: dict) -> models.PointStruct:
        """Process a single data point using the official async SDK."""
        try:
            response = await ai_client.aio.models.embed_content(
                model="models/gemini-embedding-2",
                contents=data["text_to_embed"],
                config=types.EmbedContentConfig(output_dimensionality=3072),
            )
            dense_vector = response.embeddings[0].values

            return models.PointStruct(
                id=data["point_id"],
                vector={
                    "text_dense": dense_vector,
                    "text_sparse": models.Document(
                        text=data["text_to_embed"], model="Qdrant/bm25"
                    ),
                },
                payload={
                    "text_child": data["text_child_raw"],
                    "text_parent": data["text_parent_utuh"],
                    "uu_id": data["uu_id"],
                    "status_keberlakuan": "berlaku",
                    "parent_id": data["parent_id"],
                    "bab": data["bab"],
                    "pasal": data["pasal"],
                },
            )
        except Exception as e:
            logger.error(
                f"[ERROR] Failed to process embedding for point {data['point_id']}: {e}"
            )
            raise e

    for index, batch in enumerate(batches):
        logger.info(
            f"-> Executing batch wave {index + 1}/{len(batches)} "
            f"({len(batch)} records)..."
        )

        tasks = [process_single_point(p) for p in batch]
        qdrant_points_batch = await asyncio.gather(*tasks)

        logger.info(
            "   -> Uploading "
            f"{len(qdrant_points_batch)} points "
            f"from batch wave {index + 1} to Qdrant Cloud..."
        )

        await async_qdrant_client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME, points=qdrant_points_batch
        )
        logger.info(f"   [SUCCESS] Batch wave {index + 1} safely stored in Qdrant.")

        if index < len(batches) - 1:
            logger.info(
                "-> 65-second pause to reset Google Free Tier Gemini "
                "RPM quota before next batch...\n"
            )
            await asyncio.sleep(65)

    logger.info(
        f"\n[DONE] All {len(batches)} batches successfully transferred to Qdrant Cloud!"
    )
