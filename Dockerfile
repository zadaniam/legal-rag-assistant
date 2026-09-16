# ==============================================================================
# STAGE 1: Base (Penyiapan Environment & Pemasangan Paket Secara Global)
# ==============================================================================
# Menggunakan base image uv resmi berbasis debian-slim untuk efisiensi tinggi
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base

WORKDIR /app

# Optimasi internal Python untuk lingkungan kontainer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Salin seluruh berkas konfigurasi paket dan kunci versi (lockfile)
COPY pyproject.toml uv.lock ./

# Pasang semua dependensi utama (termasuk FastAPI, LangGraph, dan Chainlit)
# Menggunakan cache mount uv agar proses build berikutnya berjalan instan
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Salin seluruh kode sumber proyek Anda ke dalam image (termasuk src/, app_chainlit.py, dll)
COPY . .

# Set variabel global agar Pydantic otomatis mendeteksi berkas .env.production
ENV APP_ENV=production

# Pastikan virtual environment bawaan uv masuk ke dalam PATH sistem kontainer
ENV PATH="/app/.venv/bin:$PATH"

# ==============================================================================
# STAGE 2: Target Runtime untuk Backend (FastAPI)
# ==============================================================================
FROM base AS backend

# Cloud Run menyuntikkan port secara dinamis melalui env $PORT.
# Kita jalankan uvicorn mengikat ke port tersebut.
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port $PORT"]

# ==============================================================================
# STAGE 3: Target Runtime untuk Frontend (Chainlit)
# ==============================================================================
FROM base AS frontend

# Jalankan Chainlit mengikat ke port dinamis Cloud Run.
# Parameter --headless wajib agar kontainer tidak mencoba membuka browser di dalam server.
CMD ["sh", "-c", "chainlit run ui/app_chainlit.py --host 0.0.0.0 --port ${PORT:-8080} --headless"]
