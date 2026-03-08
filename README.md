# Newsgroups Semantic Search System

A production-grade semantic search engine over the 20 Newsgroups corpus, featuring fuzzy clustering, a first-principles SQLite semantic cache, and a FastAPI service — built for the Trademarkia AI/ML Engineer internship assignment.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Novel Contributions](#novel-contributions)
3. [Concept Explanations](#concept-explanations)
4. [Technology Choices](#technology-choices)
5. [Project Structure](#project-structure)
6. [Quick Start](#quick-start)
7. [API Reference](#api-reference)
8. [Semantic Cache Deep Dive](#semantic-cache-deep-dive)
9. [Fuzzy Clustering Deep Dive](#fuzzy-clustering-deep-dive)
10. [Docker](#docker)
11. [Deployment](#deployment)
12. [Design Decisions](#design-decisions)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OFFLINE PIPELINE                         │
│                                                                 │
│  20 Newsgroups  →  Clean  →  Embed (MiniLM)  →  ChromaDB       │
│                                    │                            │
│                             Embeddings.npy                      │
│                                    │                            │
│                           Fuzzy C-Means (FCM)                   │
│                                    │                            │
│                    cluster_memberships.npy + cluster_meta.json  │
│                                    │                            │
│                         UMAP + Plotly → cluster_viz.html        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        ONLINE (API) PIPELINE                    │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ▼                                                          │
│  Embed Query (MiniLM, 384-dim, L2-normalised)                   │
│      │                                                          │
│      ▼                                                          │
│  Assign to nearest cluster centroid(s)                          │
│      │                                                          │
│      ▼                                                          │
│  Semantic Cache Lookup (SQLite, cluster-filtered)               │
│      │                                                          │
│   ┌──┴──────────────────────────────────────┐                  │
│   │  HIT (cosine_sim ≥ 0.82)                │  MISS            │
│   │  Return cached result instantly         │    │             │
│   └─────────────────────────────────────────┘    ▼             │
│                                           ChromaDB Search       │
│                                                  │             │
│                                           Store in Cache        │
│                                                  │             │
│                                           Return Results        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Novel Contributions

This project goes beyond the base assignment in four meaningful ways:

### 1. Production-grade Semantic Cache
The cache is not a simple dictionary. It uses:
- **SQLite persistence**: survives server restarts; atomic writes prevent corruption
- **TTL expiry**: entries expire after 24 hours to prevent stale results
- **LRU eviction**: when the cache exceeds 1000 entries, the least recently used entries are removed
- **Cluster-aware lookup**: instead of comparing against every cache entry, we first identify the query's nearest clusters and only compare within them — reducing comparisons from O(N) to O(N/C) where C is the number of clusters

### 2. Fuzzy Clustering with Justification
- Uses FCM (Fuzzy C-Means), not K-Means — documents get probability distributions, not hard labels
- Cluster count (15) is justified via Fuzzy Partition Coefficient (FPC) elbow analysis, not chosen arbitrarily
- Boundary documents are identified and exposed via the API

### 3. Interactive Cluster Visualization
- UMAP projects 384-dim embeddings to 2D, preserving global structure better than t-SNE
- Interactive Plotly HTML shows cluster groups, fuzzy boundaries, and ambiguous documents
- Marker size encodes certainty: large = confident cluster member, small = ambiguous

### 4. Automatic Cluster Summarization
- TF-IDF keyword extraction generates 3-word semantic labels per cluster (e.g. "computer graphics hardware")
- No external API needed — fully offline, deterministic, and fast

---

## Concept Explanations

### What is Semantic Search?
Traditional keyword search looks for exact word matches. Semantic search understands **meaning**. If you search for "automobile problems", it also returns documents about "car issues" because the embedding model knows these mean the same thing.

### What is a Vector Embedding?
A vector embedding converts text into a list of numbers (a vector) where **similar meaning → similar vectors**. The model `all-MiniLM-L6-v2` converts any text to a 384-dimensional vector. Documents about cars will have vectors close to each other in this 384-dimensional space.

### What is a Vector Database?
A database optimised for finding nearest neighbours in high-dimensional space. ChromaDB stores document embeddings and efficiently finds the documents most similar to a query embedding using approximate nearest neighbour (ANN) search.

### What is Fuzzy Clustering?
Standard K-Means assigns every document to exactly one cluster. Fuzzy C-Means assigns each document a **probability distribution** across all clusters. A document about gun legislation might be 50% Politics, 35% Firearms, 15% Law. This is more honest about the messy reality of natural language.

### What is a Semantic Cache?
A cache where the key is not a string (which requires exact match) but a **meaning** (which allows approximate match). Two queries with cosine similarity ≥ 0.82 are treated as equivalent and the second one gets the cached result from the first.

### Why Traditional Caching Fails for NLP
A standard dict/Redis cache uses the query string as a key. `"What is Python?"` and `"Can you explain Python?"` are two different strings → two different cache misses → double computation. A semantic cache embeds both queries and recognises they are 0.94 cosine-similar → cache hit.

---

## Technology Choices

| Component | Chosen | Why |
|-----------|--------|-----|
| Embeddings | `all-MiniLM-L6-v2` | 22M params, 384-dim, fast on CPU, MIT license, excellent semantic quality |
| Vector DB | ChromaDB | Embedded (no server), cosine distance built-in, persistent, simple API |
| Clustering | scikit-fuzzy FCM | Only Python FCM implementation; supports soft membership natively |
| API | FastAPI | Async support, auto OpenAPI docs, Pydantic validation, faster than Flask |
| Cache storage | SQLite | ACID transactions, no server, efficient partial queries, thread-safe |
| Viz reduction | UMAP | Faster than t-SNE, preserves global structure, deterministic with seed |
| Viz library | Plotly | Interactive HTML output, no JavaScript knowledge needed |

---

## Project Structure

```
newsgroups-semantic-search/
│
├── app/
│   ├── main.py                    # FastAPI app, startup lifespan
│   ├── api/
│   │   └── routes.py              # POST /query, GET /cache/stats, DELETE /cache
│   ├── services/
│   │   ├── embedding_service.py   # Singleton SentenceTransformer wrapper
│   │   ├── vector_db_service.py   # ChromaDB semantic search
│   │   ├── clustering_service.py  # Cluster assignment for query routing
│   │   └── cache_service.py       # SQLite semantic cache (TTL + LRU + clusters)
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models
│   └── utils/
│       └── __init__.py
│
├── scripts/
│   ├── prepare_data.py            # Load → clean → embed → store in ChromaDB
│   └── build_clusters.py         # FCM → UMAP → Plotly visualisation
│
├── data/                          # Created by scripts (gitignored)
│   ├── chroma_db/                 # ChromaDB persistent storage
│   ├── embeddings.npy             # Raw embeddings array (n_docs, 384)
│   ├── cluster_memberships.npy    # FCM memberships (n_docs, n_clusters)
│   ├── cluster_meta.json          # Cluster labels, FPC, boundary docs
│   └── doc_meta.json              # Document IDs, original labels, previews
│
├── cache/
│   └── cache.db                   # SQLite semantic cache (auto-created)
│
├── visualizations/
│   └── cluster_viz.html           # Interactive UMAP cluster plot
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- ~4 GB RAM (for embeddings)
- ~2 GB disk (for model weights + ChromaDB)

### Setup

```bash
# 1. Clone and enter project
git clone <your-repo-url>
cd newsgroups-semantic-search

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare data (loads dataset, generates embeddings, stores in ChromaDB)
#    ~5–10 minutes on CPU for 5000 documents
python scripts/prepare_data.py

# 5. Build clusters and visualisation
#    ~2–3 minutes on CPU
python scripts/build_clusters.py

# 6. Start the API
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive API explorer.

---

## API Reference

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I configure a Linux firewall?"}'
```

Response:
```json
{
  "query": "How do I configure a Linux firewall?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [
    {
      "document": "...",
      "label_name": "comp.os.ms-windows.misc",
      "similarity": 0.7821
    }
  ],
  "dominant_cluster": 4,
  "cluster_label": "linux network system"
}
```

### `GET /cache/stats`

```bash
curl http://localhost:8000/cache/stats
```

Response:
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405,
  "similarity_threshold": 0.82,
  "max_entries": 1000,
  "ttl_seconds": 86400
}
```

### `DELETE /cache`

```bash
curl -X DELETE http://localhost:8000/cache
```

---

## Semantic Cache Deep Dive

### How it works

1. **Embed** the incoming query → 384-dim unit vector
2. **Cluster assignment** → find top-3 nearest cluster IDs
3. **SQL query**: `SELECT * FROM cache_entries WHERE cluster_id IN (c1, c2, c3)`
4. **Cosine similarity** against each candidate (dot product of unit vectors)
5. **If best similarity ≥ 0.82** → cache hit, return stored result
6. **Else** → cache miss, run full search, `INSERT` new entry

### Similarity Threshold Analysis

| Threshold | Behaviour | Use Case |
|-----------|-----------|----------|
| 0.95 | Only near-identical queries hit. Low false-positives. Low hit rate. | High-precision production |
| 0.82 | Paraphrases hit. Good balance. | General use (default) |
| 0.70 | Topic-similar queries hit. High hit rate. Risk of off-topic results. | Exploratory/demo |

The assignment explicitly asks you to **explore this parameter** — the cache.db makes this easy to do empirically by varying the threshold and measuring hit/miss rates.

### Why Cluster-Aware Lookup Matters

For a cache with N entries and C clusters, assuming uniform distribution:
- Naive lookup: O(N) comparisons
- Cluster-filtered lookup: O(N/C) comparisons on average

With N=1000 and C=15: 1000 vs ~67 comparisons. At N=10,000 this is the difference between a fast and a slow API.

---

## Fuzzy Clustering Deep Dive

### Fuzziness Parameter m

- `m = 1.001`: approaches hard assignment (nearly K-Means)
- `m = 2.0`: standard default — good soft boundaries
- `m = 5.0`: very fuzzy — all memberships approach 1/C

We use `m = 2.0` which is the theoretically motivated default.

### Example document membership

```
Document: "Gun control legislation vote in Congress"

Cluster 2 (politics congress vote):     0.48
Cluster 7 (firearms gun legislation):   0.39
Cluster 11 (law rights amendment):      0.13
```

This is the output FCM produces. K-Means would force this into one cluster and discard the rich semantic overlap.

### Boundary Documents

A document is "boundary" if its maximum cluster membership < 0.15. These documents genuinely sit between clusters — they are the most semantically interesting because they reveal where the model is uncertain.

---

## Docker

```bash
# Build image
docker build -t newsgroups-search .

# Run (assumes data/ was prepared on host)
docker-compose up
```

The service starts on `http://localhost:8000`.

---

## Deployment (Free)

### Render

1. Push to GitHub
2. New Web Service → connect repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add a persistent disk for `/data` and `/cache`

### Railway

```bash
railway init
railway up
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| `remove=('headers','footers','quotes')` | Headers leak category labels (data leakage). Footers are boilerplate. Quotes duplicate parent posts. |
| Skip docs < 20 words | Too little signal to produce a meaningful embedding; pollutes cluster centroids |
| Normalise embeddings | Makes cosine similarity == dot product — faster, and distance is length-invariant |
| 15 clusters not 20 | FPC elbow analysis shows semantic structure stabilises around 15. Some newsgroup categories are semantically near-identical. |
| LRU over LFU | LRU adapts when query patterns change over time. LFU keeps high-frequency old entries even after they become irrelevant. |
| Lazy TTL expiry | Simpler than a background thread. Cost is amortised across requests. Avoids race conditions. |
| 1 Uvicorn worker in Docker | In-memory stats counters are not process-safe. For multi-worker, migrate stats to SQLite. |
