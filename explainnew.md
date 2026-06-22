# Newsgroups Semantic Search — feature/dict-cache Branch

## What Changed from main

This branch has three changes on top of the original project:

1. **`cache_service.py` replaced** — SQLite is gone, replaced with a plain Python in-memory dictionary
2. **`POST /query` returns `retrieval_time_ms`** — shows how long the cache lookup or vector DB search took
3. **`POST /benchmark` added** — new endpoint that runs the same query against both cache and ChromaDB and returns both timings side by side

Everything else (embedding model, ChromaDB, clustering, UMAP pipeline) is identical to `main`.

---

## Why the Cache Was Changed to a Python Dict

The original SQLite cache stored entries on disk and survived server restarts. But a cache by definition is supposed to be temporary — losing it on restart is acceptable and expected behaviour.

SQLite also had a hidden cost: **every single cache lookup triggered a DELETE query** (lazy TTL expiry) followed by a SELECT, then on a hit an UPDATE. That's three disk operations per lookup, which is why cache hits were sometimes *slower* than cache misses (ChromaDB HNSW is C++ and doesn't hit disk).

The dict cache keeps everything in RAM. Same logic, no disk I/O.

---

## How the Dict Cache Works

```
_cache = {
    0: { original_query, query_embedding, result, cluster_id,
         access_count, last_accessed, created_at },
    1: { ... },
    ...
}
```

Each entry is stored under an integer key (`_next_id` increments on every store). The embedding is stored as a raw numpy array — no serialization/deserialization needed unlike the SQLite blob approach.

### Lookup flow

```
lookup(query_embedding)
  │
  ├─ Delete expired entries (created_at older than 24h)
  │
  ├─ Loop through all entries
  │    └─ sim = dot(query_embedding, entry["query_embedding"])
  │
  ├─ IF best_sim >= 0.65 → CACHE HIT
  │    └─ update access_count + last_accessed
  │    └─ return { matched_query, similarity_score, result, dominant_cluster }
  │
  └─ ELSE → return None (cache miss)
```

### Store flow

```
store(original_query, query_embedding, result, cluster_id)
  │
  ├─ Insert new entry under _next_id
  │
  └─ IF len(_cache) > 1000 → LRU eviction
       └─ delete entry with oldest last_accessed
```

### Functions

| Function | What it does |
|----------|-------------|
| `init_cache()` | No-op — nothing to set up for a dict |
| `lookup(query_embedding)` | Find semantically similar cached query |
| `store(...)` | Save new query + result into dict |
| `get_stats()` | Return entry count, hit count, miss count, hit rate |
| `flush_cache()` | Clear the entire dict, reset all counters |

---

## Why Cache Hit Can Still Be Slower Than Cache Miss

This is a counterintuitive result you may observe:

**Cache miss** → ChromaDB HNSW search (~9ms)
ChromaDB is written in C++ and uses approximate nearest neighbor search. It does not scan all 16,781 documents — it jumps through a graph and visits only a fraction.

**Cache hit** → Python dict linear scan (~13ms with few entries)
Even with just 1–2 entries in the cache, Python has overhead: thread lock acquisition, looping, dot product per entry. As cache grows large (hundreds of entries), the scan gets slower.

**The real advantage of cache** is not raw retrieval speed — it is skipping the **embedding step (~25ms BGE inference)** and offloading pressure from ChromaDB under concurrent load. The `retrieval_time_ms` field only measures the retrieval step, not the full pipeline.

To see the true benefit, compare total end-to-end response time (including embedding) between a first query and a repeated similar query.

---

## New API Endpoints

### `POST /query` — now includes `retrieval_time_ms`

```json
{
  "query": "religious arguments for god belief"
}
```

Response:
```json
{
  "query": "religious arguments for god belief",
  "cache_hit": true,
  "matched_query": "Christianity and the existence of god",
  "similarity_score": 0.7812,
  "result": [...],
  "dominant_cluster": 7,
  "cluster_label": "religion/christian/church",
  "retrieval_time_ms": 13.92
}
```

- On **cache hit**: `retrieval_time_ms` = time taken for the dict lookup
- On **cache miss**: `retrieval_time_ms` = time taken for the ChromaDB HNSW search

### `POST /benchmark` — compare both timings for the same query

```json
{
  "query": "nasa space shuttle launch"
}
```

Response:
```json
{
  "query": "nasa space shuttle launch",
  "cache_hit": true,
  "cache_time_ms": 4.21,
  "vector_db_time_ms": 8.73,
  "cache_result": [...],
  "vector_db_result": [...]
}
```

This always runs **both** the cache lookup and the ChromaDB search regardless of cache hit, so you can directly compare the two retrieval times for the same query.

---

## What Stayed the Same

| Component | Status |
|-----------|--------|
| BGE-small-en-v1.5 embedding model | Unchanged |
| ChromaDB HNSW vector store | Unchanged |
| Two-stage UMAP pipeline (384→50→2) | Unchanged |
| Fuzzy C-Means clustering (15 clusters, FPC=0.5881) | Unchanged |
| TF-IDF cluster labels | Unchanged |
| Similarity threshold (0.65) | Unchanged |
| TTL (24h), LRU eviction (max 1000 entries) | Unchanged — just in RAM now |
| `GET /cache/stats` | Unchanged |
| `DELETE /cache` | Unchanged |

---

## Trade-offs: Dict vs SQLite

| | Python Dict (this branch) | SQLite (main branch) |
|--|--------------------------|----------------------|
| Persistence | Lost on restart | Survives restarts |
| Speed | Faster (RAM only) | Slower (disk I/O per lookup) |
| Concurrent writes | Thread lock on entire dict | SQLite WAL handles this natively |
| Filtered lookup | Full scan every time | Can filter by cluster_id with SQL index |
| Correct cache behaviour | Yes — ephemeral by design | No — persisting a cache defeats its purpose |
| Dependencies | None (stdlib only) | None (stdlib sqlite3) |

The dict version is the more honest cache implementation. SQLite made more sense if the system were storing expensive-to-recompute results (e.g., LLM responses costing $0.10 each) where persistence across restarts has real value.

---

## How to Run This Branch

```bash
git checkout feature/dict-cache
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` to see all endpoints including `/benchmark`.

The cache starts empty on every server start — which is correct cache behaviour.
