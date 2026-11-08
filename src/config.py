"""Application configuration and environment settings."""

import os
from pathlib import Path

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

app_env = os.getenv("APP_ENV", "development").lower()
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    ENVIRONMENT: str = Field(default="development", validation_alias="APP_ENV")
    LOG_LEVEL: str = Field(default="INFO")

    GEMINI_API_KEY: SecretStr = Field(...)

    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_COLLECTION_NAME: str = Field(default="legal_knowledge_base_dev")
    QDRANT_API_KEY: SecretStr | None = Field(default=None)

    POSTGRES_URL: str = Field(
        default=(
            "postgresql+psycopg://legal_admin:super_secret_password"
            "@localhost:5432/legal_assistant_db"
        )
    )

    LANGSMITH_TRACING: str = Field(default="true")
    LANGSMITH_API_KEY: SecretStr = Field(...)
    LANGSMITH_PROJECT: str = Field(default="asisten-legal-rag-dev")

    model_config = SettingsConfigDict(
        env_file=f".env.{app_env}", env_file_encoding="utf-8", extra="ignore"
    )


class PipelineSettings(BaseSettings):
    """File path configuration for the data ingestion pipeline."""

    @computed_field
    def input_pdf(self) -> Path:
        """Path to the raw legal document PDF source file."""
        return BASE_DIR / "data/raw/uu_nomor_24_tahun_2011_bpjs.pdf"

    @computed_field
    def extracted_dir(self) -> Path:
        """Directory containing preprocessed extracted text files."""
        return BASE_DIR / "data/preprocessed/extracted/uu_bpjs/"

    @computed_field
    def cleaned_file(self) -> Path:
        """Path to the cleaned combined text file."""
        return BASE_DIR / "data/preprocessed/cleaned/uu_bpjs_cleaned.txt"

    @computed_field
    def parsed_md(self) -> Path:
        """Path to the parsed Markdown output file."""
        return BASE_DIR / "data/preprocessed/parsed/uu_bpjs_parsed.md"

    @computed_field
    def files_to_ingest(self) -> list[dict[str, str]]:
        """List of documents prepared for ingestion into Qdrant."""
        return [
            {
                "path": str(BASE_DIR / "data/processed/uu_bpjs_final.md"),
                "uu_id": "uu_bpjs_2011",
            }
        ]

    model_config = SettingsConfigDict(
        env_file=f".env.{app_env}", env_file_encoding="utf-8", extra="ignore"
    )


def setup_telemetry_environment() -> None:
    """Inject tracing credentials into OS environment at application startup."""
    os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT


settings = Settings()
pipeline_settings = PipelineSettings()
