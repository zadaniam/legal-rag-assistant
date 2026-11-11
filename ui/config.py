"""Frontend configuration and environment settings."""

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

app_env = os.getenv("APP_ENV", "development").lower()


class UISettings(BaseSettings):
    """Settings specific to the Chainlit frontend service."""

    ENVIRONMENT: str = Field(default="development", validation_alias="APP_ENV")
    BACKEND_URL: str = Field(default="http://127.0.0.1:7000")

    model_config = SettingsConfigDict(
        env_file=f".env.{app_env}", env_file_encoding="utf-8", extra="ignore"
    )


ui_settings = UISettings()
