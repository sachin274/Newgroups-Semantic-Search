"""
app/main.py
============
FastAPI application entry point.

startup_event initialises the cache schema and warms up the embedding model
and cluster centroids so the first request doesn't pay the cold-start cost.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router
from app.services import cache_service, embedding_service, clustering_service
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager replaces deprecated @app.on_event('startup').
    Code before the yield runs at startup; code after runs at shutdown.
    """
    print("🚀 Starting up — initialising services …")

    # Ensure SQLite schema exists
    cache_service.init_cache()
    print("  ✓ Cache DB ready")

    # Warm up embedding model (loads ~90 MB from disk into RAM)
    embedding_service.get_model()
    print("  ✓ Embedding model loaded")

    # Warm up cluster centroids (loads embeddings.npy into RAM once)
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.npy")
    if os.path.exists(data_path):
        clustering_service._load_centroids()
        print("  ✓ Cluster centroids ready")
    else:
        print("  ⚠ embeddings.npy not found — run scripts/prepare_data.py first")

    print("  ✓ API ready — visit http://localhost:8000/docs\n")

    yield  # ← application runs here

    print("Shutting down …")


app = FastAPI(
    title="Newsgroups Semantic Search",
    description=(
        "Semantic search over 20 Newsgroups with fuzzy clustering and "
        "a first-principles SQLite semantic cache."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)


@app.get("/", tags=["health"])
async def root():
    return {
        "status": "ok",
        "message": "Newsgroups Semantic Search API",
        "docs": "/docs"
    }


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
