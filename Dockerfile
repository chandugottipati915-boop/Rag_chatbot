# One image for production: the Next.js UI, exported as static files, served by the
# FastAPI app that also answers the API. One service, one URL, first-party cookies.
#
#   docker build -t paper-reader .
#   docker run -p 8000:8000 --env-file .env -e COOKIE_SECURE=false paper-reader
#
# No PyTorch: production runs with RERANK=false and OpenAI embeddings, which keeps the
# image small and the server within a free 512 MB plan.

# ---------------------------------------------------------------- 1. build the UI
FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build                       # → /ui/out (plain HTML, JS, CSS)

# ---------------------------------------------------------------- 2. the API (+ the UI)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src/ src/
COPY migrations/ migrations/
COPY alembic.ini .
COPY data/raw/catalog.json data/raw/catalog.json
COPY --from=ui /ui/out frontend/out

# production defaults; secrets (OPENAI_API_KEY, DATABASE_URL, ...) come from the host
ENV EMBEDDING_PROVIDER=openai \
    RETRIEVAL_MODE=hybrid \
    RERANK=false \
    COOKIE_SECURE=true \
    PORT=8000

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000

# 1. bring the database schema up to date, 2. start the server.
# --proxy-headers: behind the host's proxy, see the real client IP (used by the login limiter)
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn rag.api.main:app --app-dir src --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
