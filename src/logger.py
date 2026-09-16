import sys

import loguru

from src.config import settings


def setup_logger():
    loguru.logger.remove()  # Hapus logger default

    # Jika di produksi, output berupa JSON. Jika di dev, output teks biasa yang berwarna.
    log_format = "{time} | {level} | {message}"
    if settings.ENVIRONMENT == "production":
        # Mengarahkan log ke stdout dalam format terstruktur
        loguru.logger.add(sys.stdout, serialize=True, level=settings.LOG_LEVEL)
    else:
        loguru.logger.add(sys.stdout, format=log_format, level=settings.LOG_LEVEL)


setup_logger()
logger = loguru.logger
