# STAGE 1: Base
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    PATH="/app/.venv/bin:$PATH"

RUN useradd --uid 10001 --create-home --shell /bin/bash appuser && \
    chown -R 10001:10001 /app
COPY pyproject.toml uv.lock ./

# STAGE 2: Backend
FROM base AS backend

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra backend

COPY --chown=10001:10001 src/ ./src/
USER 10001
CMD ["sh", "-c", "exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# STAGE 3: Frontend
FROM base AS frontend

ENV PYTHONPATH="/app"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra frontend

COPY --chown=10001:10001 ui/ ./ui/
USER 10001
CMD ["sh", "-c", "CHAINLIT_PORT=${PORT:-8000} exec chainlit run ui/app_chainlit.py --host 0.0.0.0 --headless"]
