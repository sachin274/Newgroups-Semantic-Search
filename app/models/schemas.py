"""
app/models/schemas.py
======================
Pydantic models for request/response validation.

FastAPI uses these for:
1. Automatic input validation (raises 422 if query is missing or wrong type)
2. Auto-generated OpenAPI docs (visible at /docs)
3. Response serialisation guarantees
"""

from pydantic import BaseModel, Field
from typing import Any


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language search query")


class SearchHit(BaseModel):
    document: str
    label_name: str
    similarity: float


class QueryResponse(BaseModel):
    query:            str
    cache_hit:        bool
    matched_query:    str | None = None
    similarity_score: float | None = None
    result:           list[SearchHit]
    dominant_cluster: int
    cluster_label:    str


class CacheStats(BaseModel):
    total_entries:       int
    hit_count:           int
    miss_count:          int
    hit_rate:            float
    similarity_threshold: float
    max_entries:         int
    ttl_seconds:         int


class FlushResponse(BaseModel):
    message: str
