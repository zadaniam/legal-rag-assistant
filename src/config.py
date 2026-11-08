# src/config.py
import os
from pathlib import Path

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 1. Deteksi lingkungan global dari OS (default ke 'development')
app_env = os.getenv("APP_ENV", "development").lower()
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    ENVIRONMENT: str = Field(default="development", validation_alias="APP_ENV")
    LOG_LEVEL: str = "INFO"

    GEMINI_API_KEY: SecretStr
    QDRANT_URL: str
    QDRANT_COLLECTION_NAME: str = "legal_knowledge_base_dev"
    QDRANT_API_KEY: SecretStr | None = None
    POSTGRES_URL: str

    LANGSMITH_TRACING: str = "true"
    LANGSMITH_API_KEY: SecretStr
    LANGSMITH_PROJECT: str = "asisten-legal-rag-dev"

    model_config = SettingsConfigDict(
        env_file=f".env.{app_env}", env_file_encoding="utf-8", extra="ignore"
    )


class PipelineSettings(BaseSettings):
    @computed_field
    def input_pdf(self) -> Path:
        return BASE_DIR / "data/raw/uu_nomor_24_tahun_2011_bpjs.pdf"

    @computed_field
    def extracted_dir(self) -> Path:
        return BASE_DIR / "data/preprocessed/extracted/uu_bpjs/"

    @computed_field
    def cleaned_file(self) -> Path:
        return BASE_DIR / "data/preprocessed/cleaned/uu_bpjs_cleaned.txt"

    @computed_field
    def parsed_md(self) -> Path:
        return BASE_DIR / "data/preprocessed/parsed/uu_bpjs_parsed.md"

    @computed_field
    def files_to_ingest(self) -> list[dict[str, str]]:
        return [
            {
                "path": str(BASE_DIR / "data/processed/uu_bpjs_final.md"),
                "uu_id": "uu_bpjs_2011",
            }
        ]

    model_config = SettingsConfigDict(
        env_file=f".env.{app_env}", env_file_encoding="utf-8", extra="ignore"
    )


def setup_telemetry_environment():
    """Menyuntikkan kredensial tracing ke environment OS secara aman saat aplikasi start."""
    os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT


settings = Settings()
pipeline_settings = PipelineSettings()
