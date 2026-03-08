"""
app/services/cache_service.py
==============================
First-principles semantic cache.

WHY NOT A DICTIONARY?
---------------------
A plain dict cache works only for byte-identical queries. "How do I install
Python?" and "What's the Python installation procedure?" are semantically
identical but would be two separate dictionary keys. In a semantic search
system, this defeats the entire purpose of the cache.

WHY SQLITE OVER A JSON FILE?
-----------------------------
- Atomic writes: JSON files can be corrupted on crash mid-write. SQLite uses
  WAL (Write-Ahead Logging) so every write is atomic.
- Efficient partial queries: "give me entries from cluster 3 ordered by
  access_time" is a SQL query. With JSON you'd load the entire file.
- Thread safety: SQLite handles concurrent reads from multiple Uvicorn workers.
- Persistence: survives server restarts. The query embeddings and their cached
  results remain valid across restarts because our embedding model is frozen.

SIMILARITY THRESHOLD TRADE-OFFS
---------------------------------
High threshold (e.g. 0.95): very conservative — only nearly-identical queries
  hit the cache. Precise but low hit rate. Good when accuracy is critical.
Low threshold (e.g. 0.70): aggressive — many queries share a cached result.
  High hit rate but risks returning a result that's semantically adjacent but
  not the best answer to this specific query. Good for exploratory systems.
We default to 0.82, which provides a reasonable balance for newsgroup queries.

HOW CLUSTER-AWARENESS SPEEDS UP LOOKUP
----------------------------------------
As the cache grows to thousands of entries, checking cosine similarity against
every entry becomes expensive. If we first assign the query to its nearest
cluster(s) and only compare against cache entries from those clusters, we
reduce the comparison set dramatically. For a 10,000-entry cache with 15
clusters, this cuts average comparisons from 10,000 to ~667.

EVICTION: LRU (Least Recently Used)
-------------------------------------
When the cache exceeds MAX_ENTRIES, we delete the entries with the oldest
last_accessed timestamp. This keeps frequently-used entries and evicts stale
ones. LFU (Least Frequently Used) is an alternative — it keeps entries with
the highest hit counts. LRU is simpler and performs well when query patterns
shift over time (LFU would keep old high-frequency entries even after they
become irrelevant).

TTL (Time To Live)
-------------------
Cache entries expire after TTL_SECONDS. In production, the underlying corpus
might be updated periodically; TTL ensures stale results don't persist
indefinitely. For the demo we default to 24 hours.
"""

import os
import json
import time
import sqlite3
import threading
import numpy as np
from contextlib import contextmanager

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DB_PATH    = os.path.join(os.path.dirname(__file__), "..", "..", "cache", "cache.db")
SIMILARITY_THRESHOLD = 0.65   # Cosine similarity required for a cache hit
MAX_ENTRIES      = 1000       # LRU eviction kicks in above this
TTL_SECONDS      = 86400      # 24 hours
TOP_K_CLUSTERS   = 3          # How many clusters to search during lookup
# ─────────────────────────────────────────────────────────────────────────────

# Thread lock for stats counters (in-memory; DB handles its own serialisation)
_stats_lock = threading.Lock()
_hit_count  = 0
_miss_count = 0


@contextmanager
def _get_conn():
    """Context manager that yields a SQLite connection and auto-commits."""
    conn = sqlite3.connect(CACHE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_cache():
    """
    Create the SQLite schema if it doesn't exist.
    Called once at application startup.

    Schema design:
    - query_embedding: BLOB (serialised float32 numpy array via np.tobytes)
    - result_json: TEXT (JSON-serialised search results)
    - cluster_id: INTEGER (dominant cluster — used to narrow lookup scope)
    - similarity_score: REAL (similarity of the original query to itself = 1.0)
    - access_count: INTEGER (for potential LFU eviction)
    - last_accessed: REAL (Unix timestamp — used for LRU eviction and TTL)
    - created_at: REAL (when the entry was inserted)
    - original_query: TEXT (stored for the 'matched_query' API field)
    """
    os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                original_query  TEXT NOT NULL,
                query_embedding BLOB NOT NULL,
                result_json     TEXT NOT NULL,
                cluster_id      INTEGER NOT NULL,
                similarity_score REAL NOT NULL,
                access_count    INTEGER DEFAULT 1,
                last_accessed   REAL NOT NULL,
                created_at      REAL NOT NULL
            )
        """)
        # Index on cluster_id so cluster-filtered lookups use an index scan
        # instead of a full table scan — critical for large caches.
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cluster ON cache_entries (cluster_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries (last_accessed)
        """)


def _deserialise_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _serialise_embedding(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def lookup(query_embedding: np.ndarray,
           cluster_ids: list[int] | None = None) -> dict | None:
    """
    Search the cache for a semantically similar past query.

    Algorithm:
    1. Delete expired entries (lazy expiry — done here, not on a timer).
    2. If cluster_ids provided, restrict candidates to those clusters.
    3. Compute cosine similarity between query_embedding and all candidates.
    4. Return the best match if its similarity exceeds SIMILARITY_THRESHOLD.

    We use lazy expiry (check on lookup rather than a background thread)
    because it's simpler, avoids race conditions, and the cost is amortised
    across requests. For high-throughput production systems you'd run a
    periodic cleanup job instead.
    """
    global _hit_count, _miss_count
    now = time.time()

    with _get_conn() as conn:
        # ── Lazy TTL expiry ────────────────────────────────────────────────
        conn.execute(
            "DELETE FROM cache_entries WHERE (? - created_at) > ?",
            (now, TTL_SECONDS)
        )

        # ── Candidate selection ────────────────────────────────────────────
        if cluster_ids:
            placeholders = ','.join('?' * len(cluster_ids))
            rows = conn.execute(
                f"SELECT * FROM cache_entries WHERE cluster_id IN ({placeholders})",
                cluster_ids
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM cache_entries").fetchall()

        if not rows:
            with _stats_lock:
                _miss_count += 1
            return None

        # ── Cosine similarity against candidates ───────────────────────────
        # All embeddings are L2-normalised, so dot product == cosine similarity.
        best_sim   = -1.0
        best_row   = None

        for row in rows:
            cached_vec = _deserialise_embedding(row["query_embedding"])
            # Dot product of two unit vectors = cosine similarity
            sim = float(np.dot(query_embedding, cached_vec))
            if sim > best_sim:
                best_sim = sim
                best_row = row

        if best_sim < SIMILARITY_THRESHOLD:
            with _stats_lock:
                _miss_count += 1
            return None

        # ── Cache hit: update access stats ────────────────────────────────
        conn.execute(
            """UPDATE cache_entries
               SET access_count = access_count + 1, last_accessed = ?
               WHERE id = ?""",
            (now, best_row["id"])
        )
        with _stats_lock:
            _hit_count += 1

        return {
            "matched_query":  best_row["original_query"],
            "similarity_score": round(best_sim, 4),
            "result":          json.loads(best_row["result_json"]),
            "dominant_cluster": best_row["cluster_id"]
        }


def store(original_query: str,
          query_embedding: np.ndarray,
          result: dict,
          cluster_id: int) -> None:
    """
    Insert a new cache entry, then evict if over capacity.

    Eviction policy: LRU — remove the entry with the oldest last_accessed
    timestamp. This means recently-used entries are always preserved, which
    matches the assumption that recently-queried topics remain relevant.
    """
    now = time.time()

    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO cache_entries
               (original_query, query_embedding, result_json, cluster_id,
                similarity_score, access_count, last_accessed, created_at)
               VALUES (?, ?, ?, ?, 1.0, 1, ?, ?)""",
            (
                original_query,
                _serialise_embedding(query_embedding),
                json.dumps(result),
                cluster_id,
                now,
                now
            )
        )

        # ── LRU Eviction ───────────────────────────────────────────────────
        count = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
        if count > MAX_ENTRIES:
            # Delete the (count - MAX_ENTRIES) least recently accessed entries
            overflow = count - MAX_ENTRIES
            conn.execute(
                """DELETE FROM cache_entries WHERE id IN (
                       SELECT id FROM cache_entries
                       ORDER BY last_accessed ASC
                       LIMIT ?
                   )""",
                (overflow,)
            )


def get_stats() -> dict:
    """Return current cache statistics."""
    global _hit_count, _miss_count

    with _get_conn() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM cache_entries"
        ).fetchone()[0]

    with _stats_lock:
        h = _hit_count
        m = _miss_count

    total_requests = h + m
    return {
        "total_entries": total,
        "hit_count":     h,
        "miss_count":    m,
        "hit_rate":      round(h / total_requests, 4) if total_requests > 0 else 0.0,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "max_entries":   MAX_ENTRIES,
        "ttl_seconds":   TTL_SECONDS
    }


def flush_cache() -> None:
    """Delete all cache entries and reset stats counters."""
    global _hit_count, _miss_count
    with _get_conn() as conn:
        conn.execute("DELETE FROM cache_entries")
    with _stats_lock:
        _hit_count  = 0
        _miss_count = 0
