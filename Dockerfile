# STAGE 1: Base
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./

# STAGE 2: Backend
FROM base AS backend
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra backend
COPY src/ ./src/
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port $PORT"]

# STAGE 3: Frontend
FROM base AS frontend
ENV PYTHONPATH="/app"
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra frontend
COPY ui/ ./ui/
CMD ["sh", "-c", "CHAINLIT_PORT=$PORT chainlit run ui/app_chainlit.py --host 0.0.0.0 --headless"]
