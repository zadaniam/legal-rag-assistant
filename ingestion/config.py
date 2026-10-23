# config.py
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# 1. Tentukan berkas .env mana yang akan dibaca berdasarkan argumen Makefile (APP_ENV)
APP_ENV = os.getenv("APP_ENV", "development")
ENV_FILE = ".env.production" if APP_ENV == "production" else ".env.development"

print(
    f"[CONFIG] Memuat konfigurasi lingkungan: {APP_ENV.upper()} dari berkas {ENV_FILE}"
)


class Settings(BaseSettings):
    ENVIRONMENT: str = APP_ENV
    LOG_LEVEL: str = "INFO"

    GEMINI_API_KEY: str

    # Jika dev: http://localhost:6333 | Jika prod: https://endpoint-qdrant-cloud
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION_NAME: str = "legal_knowledge_base"

    POSTGRES_URL: str

    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "asisten-legal-rag"

    # Konfigurasi Pydantic Settings Engine
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",  # Mengabaikan variabel ekstra di file env agar tidak crash
    )


# Ekspor objek global tunggal (Singleton)
settings = Settings()
