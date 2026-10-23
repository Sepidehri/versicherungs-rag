FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY config ./config
COPY data/documents ./data/documents
COPY data/eval ./data/eval

# Build the index at image build time so the container starts ready to answer.
# This also downloads the embedding model into the image.
RUN vrag ingest

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["vrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
