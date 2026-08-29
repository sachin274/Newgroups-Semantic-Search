"""
app/services/cache_service.py
==============================
In-memory semantic cache using a plain Python dictionary.

Each cache entry is stored as a dict value keyed by a unique integer ID.
On every lookup, we compute cosine similarity between the incoming query
embedding and all stored embeddings, returning the best match above the
threshold.

TTL:  entries older than TTL_SECONDS are deleted lazily during lookup.
LRU:  when the dict exceeds MAX_ENTRIES, the entry with the oldest
      last_accessed time is removed.
"""

import json
import time
import threading
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.65
MAX_ENTRIES          = 1000
TTL_SECONDS          = 86400   # 24 hours
# ─────────────────────────────────────────────────────────────────────────────

_lock       = threading.Lock()
_hit_count  = 0
_miss_count = 0
_next_id    = 0

# Main store: { id: { original_query, query_embedding (np.ndarray),
#                     result (list), cluster_id (int),
#                     access_count, last_accessed, created_at } }
_cache: dict[int, dict] = {}


def init_cache() -> None:
    """No-op for the dict cache — nothing to initialise on disk."""
    pass


def lookup(query_embedding: np.ndarray,
           cluster_ids: list[int] | None = None) -> dict | None:
    """
    Search the in-memory cache for a semantically similar past query.

    Steps:
    1. Delete expired entries (lazy TTL).
    2. If cluster_ids is given, only compare against entries whose cluster_id
       is in that list (cluster-narrowed scan) — otherwise scan every entry.
    3. Compute cosine similarity against the candidate entries.
    4. Return the best match if similarity >= SIMILARITY_THRESHOLD.

    Narrowing by cluster_ids trades a small recall risk (a similar cached
    query assigned to a cluster outside cluster_ids will be missed) for a
    smaller scan — no fallback to a full scan is performed on a filtered miss.
    """
    global _hit_count, _miss_count

    now = time.time()
    cluster_filter = set(cluster_ids) if cluster_ids is not None else None

    with _lock:
        # ── Lazy TTL expiry ────────────────────────────────────────────────
        expired = [k for k, v in _cache.items()
                   if (now - v["created_at"]) > TTL_SECONDS]
        for k in expired:
            del _cache[k]

        if not _cache:
            _miss_count += 1
            return None

        # ── Cosine similarity against candidate entries ─────────────────────
        best_sim = -1.0
        best_key = None

        for key, entry in _cache.items():
            if cluster_filter is not None and entry["cluster_id"] not in cluster_filter:
                continue
            sim = float(np.dot(query_embedding, entry["query_embedding"]))
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key is None:
            _miss_count += 1
            return None

        if best_sim < SIMILARITY_THRESHOLD:
            _miss_count += 1
            return None

        # ── Cache hit ─────────────────────────────────────────────────────
        _cache[best_key]["access_count"] += 1
        _cache[best_key]["last_accessed"] = now
        _hit_count += 1

        entry = _cache[best_key]
        return {
            "matched_query":    entry["original_query"],
            "similarity_score": round(best_sim, 4),
            "result":           entry["result"],
            "dominant_cluster": entry["cluster_id"]
        }


def store(original_query: str,
          query_embedding: np.ndarray,
          result: list,
          cluster_id: int) -> None:
    """
    Insert a new entry into the in-memory cache.
    Evicts the least recently used entry if over capacity.
    """
    global _next_id

    now = time.time()

    with _lock:
        _cache[_next_id] = {
            "original_query":  original_query,
            "query_embedding": query_embedding,
            "result":          result,
            "cluster_id":      cluster_id,
            "access_count":    1,
            "last_accessed":   now,
            "created_at":      now
        }
        _next_id += 1

        # ── LRU eviction ──────────────────────────────────────────────────
        if len(_cache) > MAX_ENTRIES:
            lru_key = min(_cache, key=lambda k: _cache[k]["last_accessed"])
            del _cache[lru_key]


def get_stats() -> dict:
    """Return current cache statistics."""
    with _lock:
        return {
            "total_entries":        len(_cache),
            "hit_count":            _hit_count,
            "miss_count":           _miss_count,
            "hit_rate":             round(_hit_count / (_hit_count + _miss_count), 4)
                                    if (_hit_count + _miss_count) > 0 else 0.0,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "max_entries":          MAX_ENTRIES,
            "ttl_seconds":          TTL_SECONDS
        }


def flush_cache() -> None:
    """Clear all cache entries and reset counters."""
    global _hit_count, _miss_count, _next_id
    with _lock:
        _cache.clear()
        _hit_count  = 0
        _miss_count = 0
        _next_id    = 0
