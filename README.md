# Newsgroups Semantic Search

## What This Project Is

A semantic search engine over the 20 Newsgroups corpus (~18,000 documents). You type a natural language query, the system finds the most semantically similar documents and returns them fast.

The system goes beyond a basic vector search by introducing:

- A **semantic cache** built with an in-memory Python dictionary — returns results instantly for similar past queries without hitting the vector database
- A **two-stage UMAP + Fuzzy C-Means clustering** pipeline that groups documents by topic
- **Enhanced TF-IDF cluster labeling** that generates human-readable topic names per cluster
- An **interactive Plotly visualization** of the entire document space
- A **benchmark endpoint** to compare cache retrieval speed vs vector database search speed side by side

---

## Topics / Concepts Involved

| Area                     | Concepts                                                                            |
| ------------------------ | ----------------------------------------------------------------------------------- |
| NLP / Embeddings         | Dense retrieval, sentence transformers, asymmetric query encoding, L2 normalization |
| Dimensionality Reduction | UMAP (two separate stages with different hyperparameters)                           |
| Clustering               | Fuzzy C-Means (FCM), soft assignments, membership probability matrix, FPC metric    |
| Information Retrieval    | Cosine similarity, HNSW indexing, semantic cache, cache hit/miss logic              |
| Caching Theory           | TTL expiry, LRU eviction, in-memory dict, thread safety                             |
| Backend Engineering      | FastAPI, Pydantic schemas, lifespan event hooks, singleton pattern                  |
| Text Processing          | TF-IDF, lemmatization (NLTK), custom stop words, weighted representative docs       |
| DevOps                   | Dockerfile, docker-compose, environment variable config                             |

---

## Project Structure

```
newsgroups-semantic-search/
│
├── app/                          ← FastAPI application
│   ├── main.py                   ← Entry point; lifespan startup/shutdown
│   ├── api/
│   │   └── routes.py             ← Endpoint handlers
│   ├── models/
│   │   └── schemas.py            ← Pydantic request/response models
│   └── services/
│       ├── embedding_service.py  ← BGE model singleton; query encoding
│       ├── vector_db_service.py  ← ChromaDB wrapper; HNSW search
│       ├── clustering_service.py ← Centroid lookup; cluster label resolution
│       └── cache_service.py      ← In-memory dict semantic cache (TTL + LRU)
│
├── scripts/                      ← One-time offline pipelines
│   ├── prepare_data.py           ← Load corpus → clean → embed → store in ChromaDB
│   └── build_clusters.py         ← UMAP → FCM → TF-IDF labels → visualization
│
├── test_folder/
│   ├── compare_models.py         ← BGE vs MiniLM benchmark
│   └── test_similarity.py        ← Validates cache threshold (Easy/Medium/Hard query pairs)
│
├── data/                         ← Generated artifacts
│   ├── embeddings.npy            ← (16,781 × 384) BGE embeddings
│   ├── embeddings_50d.npy        ← (16,781 × 50) UMAP-reduced for clustering
│   ├── cluster_memberships.npy   ← (16,781 × 15) FCM probability matrix
│   ├── cluster_meta.json         ← Cluster labels, FPC score, boundary doc count
│   ├── doc_meta.json             ← Document IDs, newsgroup labels, text previews
│   ├── umap_projection.npy       ← (16,781 × 2) 2D UMAP for visualization
│   └── chroma_db/                ← ChromaDB HNSW persistent index
│
├── visualizations/
│   └── cluster_viz.html          ← Interactive Plotly HTML (open in browser)
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Tech Stack

| Component                | Library / Tool                                   | Version          |
| ------------------------ | ------------------------------------------------ | ---------------- |
| Embedding model          | `sentence-transformers` (BAAI/bge-small-en-v1.5) | 2.7.0            |
| Vector database          | `chromadb`                                       | 0.5.0            |
| Clustering               | `scikit-fuzzy` (FCM)                             | 0.4.2            |
| Dimensionality reduction | `umap-learn`                                     | 0.5.6            |
| Text processing          | `scikit-learn` (TF-IDF), `nltk` (lemmatizer)     | 1.4.2 / 3.8.1    |
| Web framework            | `fastapi` + `uvicorn`                            | 0.111.0 / 0.29.0 |
| Schema validation        | `pydantic`                                       | 2.7.1            |
| Visualization            | `plotly`                                         | 5.22.0           |
| Cache storage            | Python dict (in-memory)                          | —                |
| Math / arrays            | `numpy`, `scipy`                                 | 1.26.4 / 1.13.0  |

### Why BAAI/bge-small-en-v1.5?

| Model                  | MTEB Retrieval Score | Dims | Speed |
| ---------------------- | -------------------- | ---- | ----- |
| BAAI/bge-small-en-v1.5 | **51.7**             | 384  | ~25ms |
| all-MiniLM-L6-v2       | 49.2                 | 384  | ~20ms |
| all-mpnet-base-v2      | 57.0                 | 768  | ~80ms |

BGE wins the accuracy/speed trade-off. It also uses **asymmetric encoding**: queries get an instruction prefix (`"Represent this sentence for searching relevant passages: "`), documents do not. This tells the model to treat the input as a search query rather than a passage, improving retrieval quality.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        CLIENT                            │
│             POST /query  {query: "..."}                  │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                 FastAPI  (app/main.py)                    │
│            routes.py → query_handler()                   │
└──────┬──────────────────┬──────────────────┬─────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
EmbeddingService   ClusteringService    CacheService
(BGE model)        (centroid lookup)    (Python dict)
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │ (on cache miss only)
                  VectorDBService
                  (ChromaDB HNSW)
```

### Layered view

```
API Layer        FastAPI routes + Pydantic schemas
     │
Service Layer    EmbeddingService | ClusteringService | VectorDBService
     │
Cache Layer      In-memory Python dict (TTL + LRU eviction)
     │
Data Layer       ChromaDB (HNSW)  +  .npy / .json pre-computed artifacts
```

---

## Two Separate Pipelines

### OFFLINE PIPELINE — run once before starting the server

This pipeline prepares all the data the server needs. It runs two scripts sequentially.

```
─── prepare_data.py  (~20 min on CPU) ───────────────────────────────

20 Newsgroups corpus (raw ~20,000 docs from scikit-learn)
  │
  ▼
Clean text
  - Remove email headers, footers, quoted replies, URLs
  - Keep only docs with ≥ 20 words after cleaning
  - Result: 16,781 documents

  │
  ▼
Embed with BGE-small-en-v1.5
  - No instruction prefix on documents (BGE convention)
  - Batch size: 64
  - L2-normalize each vector so dot product = cosine similarity
  - Output: embeddings.npy  (16,781 × 384)

  │
  ▼
Store in ChromaDB
  - Cosine distance metric
  - One entry per document with metadata (newsgroup label)
  - Persisted to data/chroma_db/


─── build_clusters.py  (~5 min on CPU) ──────────────────────────────

embeddings.npy  (16,781 × 384)
  │
  ▼
UMAP Stage 1 — reduce for clustering
  - 384-dim → 50-dim
  - Parameters: n_neighbors=15, min_dist=0.0
  - Why 50 dims? Euclidean distance is nearly uniform in 384 dimensions
    (curse of dimensionality) — FCM fails to find structure there.
    50-dim restores meaningful geometry.
  - Output: embeddings_50d.npy

  │
  ▼
Fuzzy C-Means clustering on 50-dim embeddings
  - C = 15 clusters, fuzziness m = 2.0
  - Each document gets a probability over all 15 clusters (not hard assignment)
  - Output: cluster_memberships.npy  (16,781 × 15)
  - Quality: FPC = 0.5881  (random baseline for 15 clusters = 0.067)

  │
  ▼
Extract cluster labels (Enhanced TF-IDF)
  - Lemmatize tokens: "christians" → "christian"
  - Remove Usenet-specific stop words: "don", "just", "think", "know"…
  - Weight TF-IDF scores by membership probability
  - Use top-30 highest-membership docs per cluster
  - Output: 3-word labels e.g. "space/launch/nasa", "religion/christian/church"
  - Stored in: cluster_meta.json

  │
  ▼
UMAP Stage 2 — reduce for visualization (independent of Stage 1)
  - 384-dim → 2-dim
  - Parameters: n_neighbors=30, min_dist=0.1
  - Output: umap_projection.npy

  │
  ▼
Plotly interactive visualization
  - One point per document
  - Color = dominant cluster
  - Size = membership certainty
  - White × markers = boundary documents (max membership < 0.45)
  - Hover shows top-2 cluster memberships
  - Output: visualizations/cluster_viz.html
```

### ONLINE PIPELINE — per request

```
POST /query  { "query": "nasa space shuttle launch" }
  │
  ▼
Embed query  (~25ms)
  - Prepend instruction prefix
  - BGE inference → 384-dim vector
  - L2-normalize

  │
  ▼
Assign to clusters  (~1ms)
  - Dot product against pre-computed cluster centroids
  - Identify dominant cluster + top-3 similar clusters

  │
  ▼
Semantic cache lookup  (~1–5ms)
  - Lazy TTL: delete entries older than 24h
  - Loop through all cache entries, compute cosine similarity
  - IF best similarity ≥ 0.65:
      ── CACHE HIT ──────────────────────────────────
        Update access_count + last_accessed
        Return cached result immediately
        retrieval_time_ms = dict lookup time
      ────────────────────────────────────────────────
  - ELSE: continue to ChromaDB

  │  (cache miss path only)
  ▼
ChromaDB HNSW search  (~50–100ms)
  - Query HNSW index: top-5 most similar documents
  - Convert distance → similarity: sim = 1.0 - distance
  - retrieval_time_ms = ChromaDB search time

  │
  ▼
Store in cache
  - Insert into dict under next available integer key
  - LRU eviction: if entries > 1000, delete entry with oldest last_accessed

  │
  ▼
Return response
  {
    cache_hit, matched_query, similarity_score,
    result, dominant_cluster, cluster_label,
    retrieval_time_ms
  }
```

---

## The Semantic Cache

### What it is

A regular Python dictionary where each value stores a past query's embedding and results. When a new query arrives, it is compared against every stored embedding using cosine similarity. If any stored embedding is similar enough (≥ 0.65), the cached result is returned directly — no ChromaDB search needed.

This is called a **semantic cache** because it matches by meaning, not by exact string. "NASA rocket launch" and "space shuttle takeoff" can be similar enough to share a cached result.

### Internal structure

```python
_cache = {
    0: {
        "original_query":  "nasa space shuttle launch",
        "query_embedding": np.array([...], dtype=float32),  # 384 numbers
        "result":          [ {document, label_name, similarity}, ... ],
        "cluster_id":      4,
        "access_count":    3,
        "last_accessed":   1719043200.0,   # Unix timestamp
        "created_at":      1719040000.0
    },
    1: { ... },
    ...
}
```

### Similarity threshold (0.65)

How similar does a new query need to be to get a cache hit?

```
≥ 0.90 — near-identical phrasing only     → too strict, low hit rate
≥ 0.75 — clear paraphrases               → good accuracy
≥ 0.65 — topic-similar queries  ← chosen
≤ 0.55 — false positives start appearing  → too loose
```

Calibrated empirically by testing 20 query pairs across Easy / Medium / Hard / Trick difficulty levels.

### TTL — Time To Live

Entries are deleted if they are older than 24 hours. This is done **lazily** — the check happens at lookup time, not on a background timer. Simple, no race conditions.

### LRU eviction

When the cache grows beyond 1000 entries, the entry with the oldest `last_accessed` timestamp is deleted. This keeps recently-used entries and removes ones that haven't been hit in a while.

### Thread safety

A `threading.Lock()` wraps all read and write operations on `_cache`. This ensures two concurrent requests don't corrupt the dict simultaneously.

### Cache functions

| Function                                      | What it does                                        |
| --------------------------------------------- | --------------------------------------------------- |
| `init_cache()`                                | No-op — nothing to initialize for an in-memory dict |
| `lookup(query_embedding)`                     | Scan dict, return best match above threshold        |
| `store(query, embedding, result, cluster_id)` | Insert new entry, evict if over capacity            |
| `get_stats()`                                 | Return entry count, hit count, miss count, hit rate |
| `flush_cache()`                               | Clear entire dict, reset all counters to zero       |

---

## Key Algorithms Explained

### Fuzzy C-Means (FCM) — why soft clustering

Standard K-Means assigns each document to exactly one cluster. FCM gives every document a **probability distribution** over all clusters. For example:

```
Document about "space medicine":
  cluster 4 (space/launch/nasa)  → 0.55
  cluster 9 (medicine/health)    → 0.30
  cluster 2 (science/research)   → 0.15
```

This captures natural ambiguity — newsgroup posts frequently span multiple topics.

**FPC (Fuzzy Partition Coefficient)** measures quality:

- Random baseline for 15 clusters: 0.067
- Achieved: **0.5881** — strong semantic structure found

### Two-Stage UMAP — why two separate reductions

Running FCM directly on 384-dim embeddings gives FPC ≈ 0.07. The reason: in 384 dimensions, all points are roughly equidistant from each other (curse of dimensionality). FCM can't find meaningful clusters.

**Stage 1** (for clustering): compress 384-dim → 50-dim with `min_dist=0.0` (pack similar points tightly). FCM on 50-dim achieves FPC = 0.5881 — an 8× improvement.

**Stage 2** (for visualization): compress 384-dim → 2-dim with `min_dist=0.1` (balance local and global structure for readability). Used only for the Plotly HTML — independent of Stage 1.

### Cosine similarity via dot product

All embeddings are L2-normalized at generation time, meaning their length is exactly 1.0. For two unit-length vectors:

```
cosine_similarity(a, b) = dot(a, b)
```

This means similarity comparisons are just a single numpy dot product — fast and simple.

---

## API Reference

| Method   | Endpoint       | Description                                               |
| -------- | -------------- | --------------------------------------------------------- |
| `POST`   | `/query`       | Main search — returns top documents + `retrieval_time_ms` |
| `POST`   | `/benchmark`   | Runs both cache and ChromaDB, returns both timings        |
| `GET`    | `/cache/stats` | Returns hit rate, entry count, TTL, threshold             |
| `DELETE` | `/cache`       | Flushes the entire cache                                  |

### POST /query

Request:

```json
{ "query": "nasa space shuttle launch" }
```

Response (cache miss — first time):

```json
{
  "query": "nasa space shuttle launch",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [
    {
      "document": "The shuttle program...",
      "label_name": "sci.space",
      "similarity": 0.89
    }
  ],
  "dominant_cluster": 4,
  "cluster_label": "space/launch/nasa",
  "retrieval_time_ms": 9.0
}
```

Response (cache hit — similar query asked later):

```json
{
  "query": "space shuttle takeoff nasa",
  "cache_hit": true,
  "matched_query": "nasa space shuttle launch",
  "similarity_score": 0.81,
  "result": [ ... ],
  "dominant_cluster": 4,
  "cluster_label": "space/launch/nasa",
  "retrieval_time_ms": 3.2
}
```

### POST /benchmark

Runs both cache lookup and ChromaDB search for the same query and returns both timings. The cache lookup and vector DB search both execute regardless of whether there is a cache hit.

Request:

```json
{ "query": "nasa space shuttle launch" }
```

Response:

```json
{
  "query": "nasa space shuttle launch",
  "cache_hit": true,
  "cache_time_ms": 4.21,
  "vector_db_time_ms": 8.73,
  "cache_result": [ ... ],
  "vector_db_result": [ ... ]
}
```

---

## Performance Characteristics

| Operation                  | Typical Time |
| -------------------------- | ------------ |
| Query embedding (BGE)      | ~25ms        |
| Cluster assignment         | ~1ms         |
| Cache lookup (dict scan)   | ~1–5ms       |
| ChromaDB HNSW search       | ~50–100ms    |
| Full pipeline — cache miss | ~75–130ms    |
| Full pipeline — cache hit  | ~26–30ms     |

**Why the full pipeline is faster on cache hit**: the cache saves the ChromaDB search (~50–100ms). The embedding step (~25ms) runs in both paths. So the real saving is the vector DB search, not just the retrieval step alone.

---

## Data Statistics

| Metric                                          | Value             |
| ----------------------------------------------- | ----------------- |
| Raw corpus size                                 | ~20,000 documents |
| After cleaning                                  | 16,781 documents  |
| Embedding dimensions                            | 384               |
| Number of clusters                              | 15                |
| Well-assigned documents (max membership > 0.45) | 14,348 (85.5%)    |
| Boundary documents (max membership < 0.45)      | 2,433 (14.5%)     |
| FPC score                                       | 0.5881            |
| Cache similarity threshold                      | 0.65              |
| Max cache entries                               | 1,000             |
| Cache TTL                                       | 24 hours          |

---

## How to Run

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Build data (run once — ~25 min total)
python scripts/prepare_data.py     # ~20 min
python scripts/build_clusters.py   # ~5 min

# 3. Start server
uvicorn app.main:app --reload

# 4. Open API docs
# http://localhost:8000/docs

# 5. Open cluster visualization
# open visualizations/cluster_viz.html in your browser
```

Or with Docker:

```bash
docker-compose up
```

The cache starts empty on every server start. This is correct behaviour — a cache is ephemeral by design.
