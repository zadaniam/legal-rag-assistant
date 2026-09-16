# src/config.py
import os

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# 1. Deteksi lingkungan global dari OS (default ke 'development')
app_env = os.getenv("APP_ENV", "development").lower()


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    GEMINI_API_KEY: SecretStr
    QDRANT_URL: str
    # API Key dibuat opsional agar tidak crash di lokal development
    QDRANT_API_KEY: SecretStr | None = None
    POSTGRES_URL: str

    LANGSMITH_TRACING: str = "true"
    LANGSMITH_API_KEY: SecretStr
    LANGSMITH_PROJECT: str = "asisten-legal-rag"

    # STRANDAR INDUSTRI: Membaca file secara dinamis berdasarkan nilai app_env
    model_config = SettingsConfigDict(
        env_file=f".env.{app_env}",
        env_file_encoding="utf-8",
        extra="ignore",  # Mengabaikan variabel tambahan seperti POSTGRES_PASSWORD agar tidak error
    )


settings = Settings()

# === KUNCI CLEAN CODE: Suntikkan ke OS secara otomatis di sini ===
os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING
os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY.get_secret_value()
os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
