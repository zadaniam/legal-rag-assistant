"""Application logging configuration using Loguru."""

import sys

import loguru

from src.config import settings


def setup_logger() -> None:
    """Configure Loguru output format based on environment.

    Production uses structured JSON output for log aggregation. Development
    uses colored plain text for easier local debugging.
    """
    loguru.logger.remove()

    log_format = "{time} | {level} | {message}"
    if settings.ENVIRONMENT == "production":
        loguru.logger.add(sys.stdout, serialize=True, level=settings.LOG_LEVEL)
    else:
        loguru.logger.add(sys.stdout, format=log_format, level=settings.LOG_LEVEL)


setup_logger()
logger = loguru.logger
