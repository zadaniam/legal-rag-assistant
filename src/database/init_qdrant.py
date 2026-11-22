"""Qdrant hybrid search collection initialization."""

import asyncio

from src.config import settings
from src.database.connection import (
    close_database_connections,
    get_qdrant_client,
    init_qdrant_client,
)
from src.database.vector_schema import QdrantCollectionSchema
from src.logger import logger


async def init_qdrant_collections() -> None:
    """Create and configure Qdrant collections for hybrid search."""
    collection_name = settings.QDRANT_COLLECTION_NAME
    logger.info(
        f"Checking collection '{collection_name}' in Qdrant "
        f"({settings.ENVIRONMENT.upper()})..."
    )

    client = get_qdrant_client()
    collections_response = await client.get_collections()
    existing_collections = [c.name for c in collections_response.collections]

    if collection_name in existing_collections:
        if settings.ENVIRONMENT.lower() in {"dev", "development", "local"}:
            logger.warning(
                f"Legacy collection '{collection_name}' detected in DEV mode. "
                "Deleting for hybrid search migration..."
            )
            try:
                await client.delete_collection(collection_name=collection_name)
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

    await client.create_collection(
        collection_name=collection_name,
        vectors_config=QdrantCollectionSchema.vectors_config(),
        sparse_vectors_config=QdrantCollectionSchema.sparse_vectors_config(),
    )

    for field_name, field_schema in QdrantCollectionSchema.payload_indexes():
        await client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )

    logger.info(f"Collection '{collection_name}' configured successfully.")


async def run_qdrant_initialization() -> None:
    """Run full Qdrant initialization sequence."""
    try:
        logger.info(
            f"Starting Qdrant hybrid initialization [{settings.ENVIRONMENT.upper()}]..."
        )
        await init_qdrant_client()
        await init_qdrant_collections()

        logger.info("Qdrant hybrid configured successfully.")
    except Exception as e:
        logger.error(f"Qdrant initialization failed: {e}")
        raise
    finally:
        await close_database_connections()


if __name__ == "__main__":
    asyncio.run(run_qdrant_initialization())
