"""Ingestion pipeline for loading processed legal documents into Qdrant."""

import asyncio
import os
from typing import cast

from src.config import pipeline_settings, settings
from src.data_pipeline.ingestion.parser_rag import parse_markdown_to_object
from src.data_pipeline.ingestion.qdrant_service import prepare_and_upload_to_qdrant
from src.data_pipeline.ingestion.splitter import split_laws_into_hierarchical_points
from src.database.connection import async_qdrant_client


async def run_ingest() -> None:
    """Execute the full ingestion pipeline: parse -> split -> embed -> upload."""
    print(f"[START] Ingestion pipeline running in mode: {settings.ENVIRONMENT.upper()}")

    files_to_ingest: list[dict[str, str]] = cast(
        list[dict[str, str]], pipeline_settings.files_to_ingest
    )

    try:
        for doc in files_to_ingest:
            target_file = doc["path"]
            target_id = doc["uu_id"]

            if not os.path.exists(target_file):
                print(f"[SKIP] File not found, skipping: {target_file}")
                continue

            print(
                f"\n--- Processing Document: "
                f"{os.path.basename(target_file)} ({target_id}) ---"
            )

            parsed_laws = parse_markdown_to_object(target_file, uu_id=target_id)
            prepared_points = split_laws_into_hierarchical_points(parsed_laws)

            print(
                f"-> Running Gemini embedding and uploading "
                f"{len(prepared_points)} records to Qdrant..."
            )
            await prepare_and_upload_to_qdrant(prepared_points)

            print(f"[SUCCESS] Legal data for {target_id} loaded into the database!")

    except Exception as e:
        print(f"[ERROR] Mass ingestion process failed: {e}")
        raise e

    finally:
        if async_qdrant_client is not None:
            await async_qdrant_client.close()
            print("Async Qdrant Client connection closed cleanly.")
        else:
            print("Async Qdrant client was already uninitialized or None.")


if __name__ == "__main__":
    asyncio.run(run_ingest())
