# sec-rag API service.
#
# The image contains code only. The corpus (chunk store + Qdrant local DB)
# is mounted at runtime — build once, re-ingest without rebuilding:
#
#   docker build -t sec-rag .
#   docker run -p 8000:8000 \
#     -e OPENAI_API_KEY -e COHERE_API_KEY -e SECRAG_API_KEYS=changeme \
#     -v ./data/store:/app/data/store \
#     -v ./qdrant_db:/app/qdrant_db \
#     sec-rag
#
# Note: Qdrant local mode takes an exclusive file lock — one container per
# qdrant_db volume. Move to a Qdrant server for multi-replica deployments.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install .

RUN useradd --create-home appuser \
    && mkdir -p /app/data/store /app/qdrant_db /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; \
    r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); \
    sys.exit(0 if r.status == 200 else 1)"

CMD ["uvicorn", "sec_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
