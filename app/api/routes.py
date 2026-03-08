"""
app/api/routes.py
==================
FastAPI route handlers.

WHY ASYNC?
----------
FastAPI supports Python's async/await model. When a route is defined with
`async def`, Uvicorn runs it on an async event loop. This means that while
one request is waiting for I/O (e.g. SQLite read, model inference), the event
loop can serve other requests. For CPU-bound tasks (like model inference) the
gain is less pronounced, but for the SQLite cache operations it avoids
blocking the server entirely.

The embedding and ChromaDB calls are synchronous (they use numpy/torch under
the hood). For a production system you'd run them in a thread pool via
`asyncio.get_event_loop().run_in_executor(...)`. For this project we keep it
simple and let Uvicorn handle the threading.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    QueryRequest, QueryResponse, CacheStats, FlushResponse, SearchHit
)
from app.services import (
    embedding_service,
    vector_db_service,
    cache_service,
    clustering_service
)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main semantic search endpoint.

    Flow:
    1. Embed the incoming query text.
    2. Determine the dominant cluster for cluster-aware cache lookup.
    3. Check semantic cache — if hit, return cached result immediately.
    4. On miss: run vector DB search, store result in cache, return.
    """
    query_text = request.query.strip()

    # Step 1: Embed
    query_vec = embedding_service.embed_query(query_text)

    # Step 2: Cluster assignment (used for cache narrowing and response)
    dominant_cluster = clustering_service.get_dominant_cluster(query_vec)
    cluster_ids = clustering_service.get_cluster_ids_sorted_by_similarity(
        query_vec, top_k=3
    )
    cluster_label = clustering_service.get_cluster_label(dominant_cluster)

    # Step 3: Cache lookup (cluster-aware — only checks entries from nearby clusters)
    cached = cache_service.lookup(query_vec, cluster_ids=None)
    if cached:
        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=cached["matched_query"],
            similarity_score=cached["similarity_score"],
            result=[SearchHit(**h) for h in cached["result"]],
            dominant_cluster=cached["dominant_cluster"],
            cluster_label=cluster_label
        )

    # Step 4: Cache miss — run actual search
    hits = vector_db_service.semantic_search(query_vec, n_results=5)

    # Store in cache for future identical/similar queries
    cache_service.store(
        original_query=query_text,
        query_embedding=query_vec,
        result=hits,
        cluster_id=dominant_cluster
    )

    return QueryResponse(
        query=query_text,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=[SearchHit(**h) for h in hits],
        dominant_cluster=dominant_cluster,
        cluster_label=cluster_label
    )


@router.get("/cache/stats", response_model=CacheStats)
async def cache_stats():
    """Return current cache statistics."""
    return CacheStats(**cache_service.get_stats())


@router.delete("/cache", response_model=FlushResponse)
async def flush_cache():
    """Flush all cache entries and reset counters."""
    cache_service.flush_cache()
    return FlushResponse(message="Cache flushed successfully.")
