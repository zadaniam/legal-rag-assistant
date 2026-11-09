"""Qdrant hybrid search collection initialization."""

import asyncio

from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    SparseVectorParams,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

from src.config import settings
from src.database.connection import async_qdrant_client
from src.logger import logger


async def init_qdrant_collections() -> None:
    """Create and configure Qdrant collections for hybrid search."""
    collection_name = settings.QDRANT_COLLECTION_NAME
    logger.info(
        f"Checking collection '{collection_name}' in Qdrant "
        f"({settings.ENVIRONMENT.upper()})..."
    )

    if async_qdrant_client is None:
        msg = "Qdrant client is not configured"
        raise RuntimeError(msg)
    collections_response = await async_qdrant_client.get_collections()
    existing_collections = [c.name for c in collections_response.collections]
    if collection_name in existing_collections:
        if settings.ENVIRONMENT.lower() in {"dev", "development", "local"}:
            logger.warning(
                f"Legacy collection '{collection_name}' detected in DEV mode. "
                "Deleting for hybrid search migration..."
            )
            try:
                await async_qdrant_client.delete_collection(
                    collection_name=collection_name
                )
                logger.info(f"Legacy collection '{collection_name}' deleted.")
            except Exception as e:
                logger.error(f"Failed to delete legacy collection: {e}")
                raise
        else:
            logger.error(
                f"DANGER: Collection '{collection_name}' already exists in "
                f"{settings.ENVIRONMENT.upper()} environment! "
                "Process halted to prevent production data loss. "
                "Perform manual migration or rename the collection if "
                "schema changes are needed."
            )
            raise ValueError(
                f"Overwriting active collection in {settings.ENVIRONMENT} is prohibited"
            )

    logger.info(
        f"Creating new collection '{collection_name}' with hybrid "
        "architecture (dense + sparse/BM25)..."
    )

    await async_qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text_dense": VectorParams(
                size=3072,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "text_sparse": SparseVectorParams(modifier=Modifier.IDF)
        },
    )

    await async_qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="text_child",
        field_schema=TextIndexParams(
            type=TextIndexType.TEXT,
            tokenizer=TokenizerType.WORD,
            lowercase=True,
        ),
    )

    await async_qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="status_keberlakuan",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    await async_qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="uu_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    logger.info(f"Collection '{collection_name}' configured successfully.")


async def run_qdrant_initialization() -> None:
    """Run full Qdrant initialization sequence."""
    try:
        logger.info(
            f"Starting Qdrant hybrid initialization [{settings.ENVIRONMENT.upper()}]..."
        )

        await init_qdrant_collections()

        logger.info("Qdrant hybrid configured successfully.")
    except Exception as e:
        logger.error(f"Qdrant initialization failed: {e}")
        raise
    finally:
        if async_qdrant_client is not None:
            logger.info("Closing async Qdrant client...")
            await async_qdrant_client.close()
            logger.info("Async Qdrant client closed cleanly.")


if __name__ == "__main__":
    asyncio.run(run_qdrant_initialization())
