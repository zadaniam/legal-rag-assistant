"""Centralized vector database schema and metadata payload definitions for Qdrant."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import models


class VectorName(StrEnum):
    """Named vector identifiers configured in the Qdrant collection."""

    DENSE = "text_dense"
    SPARSE = "text_sparse"


class PayloadField(StrEnum):
    """Canonical payload field keys stored in Qdrant points."""

    TEXT_CHILD = "text_child"
    TEXT_PARENT = "text_parent"
    UU_ID = "uu_id"
    STATUS_KEBERLAKUAN = "status_keberlakuan"
    PARENT_ID = "parent_id"
    BAB = "bab"
    PASAL = "pasal"


class LegalStatus(StrEnum):
    """Valid legal validity status values for filtering documents."""

    BERLAKU = "berlaku"
    DICABUT = "dicabut"
    SEMUA = "Semua"


class EmbeddingSpec:
    """Embedding model and vector dimension specifications."""

    DENSE_MODEL: str = "models/gemini-embedding-2"
    DENSE_DIMENSION: int = 3072
    DENSE_DISTANCE: models.Distance = models.Distance.COSINE

    SPARSE_MODEL: str = "Qdrant/bm25"
    SPARSE_MODIFIER: models.Modifier = models.Modifier.IDF


class LegalChunkPayload(BaseModel):
    """Validated metadata payload schema stored in Qdrant for each legal chunk.

    Maintains backward compatibility with existing Qdrant collections by preserving
    exact field names, providing safe defaults for legacy partial payloads, and
    allowing extra legacy fields during deserialization.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    text_child: str = Field(
        default="", description="Child chunk text indexed for BM25/text search."
    )
    text_parent: str = Field(
        default="", description="Full parent article text returned to LLM."
    )
    uu_id: str = Field(
        default="", description="Identifier of the source law (e.g., uu_bpjs_2011)."
    )
    status_keberlakuan: str = Field(
        default=LegalStatus.BERLAKU.value,
        description="Legal validity status of the article.",
    )
    parent_id: str = Field(
        default="", description="UUID grouping child chunks under one parent article."
    )
    bab: str = Field(
        default="", description="Chapter (BAB) metadata of the legal article."
    )
    pasal: str = Field(
        default="", description="Article (Pasal) metadata of the legal article."
    )

    def to_qdrant_payload(self) -> dict[str, Any]:
        """Serialize payload into a plain dictionary compatible with Qdrant SDK."""
        return self.model_dump()


class PreparedVectorPoint(BaseModel):
    """Validated record produced by splitter before embedding and upload."""

    model_config = ConfigDict(extra="forbid")

    point_id: str = Field(..., description="Unique UUID for the Qdrant point.")
    text_to_embed: str = Field(
        ..., description="Contextualized text sent to embedding models."
    )
    payload: LegalChunkPayload = Field(
        ..., description="Structured Qdrant payload metadata."
    )


class QdrantCollectionSchema:
    """Factory methods for Qdrant collection vectors and payload index definitions."""

    @staticmethod
    def vectors_config() -> dict[str, models.VectorParams]:
        """Return dense vector configuration matching Qdrant collection setup."""
        return {
            VectorName.DENSE.value: models.VectorParams(
                size=EmbeddingSpec.DENSE_DIMENSION,
                distance=EmbeddingSpec.DENSE_DISTANCE,
            )
        }

    @staticmethod
    def sparse_vectors_config() -> dict[str, models.SparseVectorParams]:
        """Return sparse vector configuration matching Qdrant collection setup."""
        return {
            VectorName.SPARSE.value: models.SparseVectorParams(
                modifier=EmbeddingSpec.SPARSE_MODIFIER
            )
        }

    @staticmethod
    def payload_indexes() -> list[
        tuple[str, models.PayloadSchemaType | models.TextIndexParams]
    ]:
        """Return ordered (field_name, field_schema) pairs for Qdrant indexes."""
        return [
            (
                PayloadField.TEXT_CHILD.value,
                models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    lowercase=True,
                ),
            ),
            (
                PayloadField.STATUS_KEBERLAKUAN.value,
                models.PayloadSchemaType.KEYWORD,
            ),
            (
                PayloadField.UU_ID.value,
                models.PayloadSchemaType.KEYWORD,
            ),
        ]
