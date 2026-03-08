# Dockerfile
# ===========
# Multi-stage build not used here because the model weights (~90 MB) and
# ChromaDB data need to be baked in or mounted at runtime. We keep it simple.

FROM python:3.11-slim

# System deps for numpy/scipy/umap native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer caching — only rebuilds
# this layer when requirements.txt changes, not on every code change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY scripts/ ./scripts/

# Pre-create directories for mounted volumes
RUN mkdir -p data cache visualizations

# Expose the Uvicorn port
EXPOSE 8000

# Health check so orchestrators know when the service is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the API
# --host 0.0.0.0 binds to all interfaces (required inside Docker)
# --workers 1 because our in-memory stats counters are not shared across
#   processes. For multi-worker deployments, move stats to SQLite.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
