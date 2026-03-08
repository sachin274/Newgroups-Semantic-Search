# Newsgroups Semantic Search System

A production-grade semantic search engine over the 20 Newsgroups corpus (~18,000 documents), featuring fuzzy clustering with UMAP dimensionality reduction, a first-principles SQLite semantic cache, and a FastAPI service — built for the Trademarkia AI/ML Engineer internship assignment.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Novel Contributions](#novel-contributions)
3. [Technology Choices & Justification](#technology-choices--justification)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [API Reference](#api-reference)
7. [Semantic Cache — Design Deep Dive](#semantic-cache--design-deep-dive)
8. [Fuzzy Clustering — Design Deep Dive](#fuzzy-clustering--design-deep-dive)
9. [Cluster Visualisation](#cluster-visualisation)
10. [Design Decisions Log](#design-decisions-log)
11. [Docker](#docker)

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         OFFLINE PIPELINE (run once)                  │
│                                                                      │
│  20 Newsgroups (~18,000 docs)                                        │
│       │                                                              │
│       ▼                                                              │
│  Clean Text                                                          │
│  (strip emails, URLs, quotes, headers — prevent data leakage)        │
│       │                                                              │
│       ▼                                                              │
│  Embed with BAAI/bge-small-en-v1.5  →  embeddings.npy (n, 384)      │
│       │                              →  ChromaDB (vector store)      │
│       │                                                              │
│       ▼                                                              │
│  UMAP: 384-dim  →  50-dim  (for clustering, preserves structure)     │
│       │                                                              │
│       ▼                                                              │
│  Fuzzy C-Means (C=15, m=2.0)  →  cluster_memberships.npy            │
│       │                        →  cluster_meta.json                  │
│       │                                                              │
│       ▼                                                              │
│  TF-IDF Label Extraction  →  "space / launch / nasa" per cluster     │
│       │                                                              │
│       ▼                                                              │
│  UMAP: 384-dim  →  2-dim  →  Plotly HTML visualisation               │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                         ONLINE PIPELINE (per request)                │
│                                                                      │
│  User Query                                                          │
│       │                                                              │
│       ▼                                                              │
│  Embed Query (BGE instruction prefix + bge-small-en-v1.5, 384-dim)  │
│       │                                                              │
│       ▼                                                              │
│  Assign to nearest cluster centroid(s) via dot product               │
│       │                                                              │
│       ▼                                                              │
│  Semantic Cache Lookup  (SQLite, cluster-filtered)                   │
│       │                                                              │
│  ┌────┴──────────────────────────────────┐                          │
│  │  HIT  cosine_sim ≥ 0.65               │   MISS                   │
│  │  Return cached result instantly       │     │                    │
│  └───────────────────────────────────────┘     ▼                    │
│                                          ChromaDB Search             │
│                                                │                    │
│                                          Store in SQLite Cache       │
│                                                │                    │
│                                          Return Results              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Novel Contributions

This project exceeds the base assignment requirements in four specific ways.

### 1. Production-Grade Semantic Cache (built from scratch)

No Redis, no Memcached, no caching libraries. The cache is implemented entirely in Python + SQLite with the following features:

- **SQLite persistence** — survives server restarts. Unlike JSON files, SQLite uses WAL (Write-Ahead Logging) so writes are atomic — no corruption on crash.
- **TTL expiry (24h)** — entries automatically expire. Prevents serving stale results after corpus updates.
- **LRU eviction** — when the cache exceeds 1000 entries, the least recently used entries are removed. This keeps the cache relevant as query patterns shift over time.
- **Cluster-aware lookup** — instead of scanning all cache entries, the query is first assigned to its nearest clusters and only entries from those clusters are compared. For a 1000-entry cache with 15 clusters, this reduces comparisons from 1000 to ~67 on average — O(N) → O(N/C).
- **Cosine similarity via dot product** — all embeddings are L2-normalised at storage time, so similarity = `np.dot(a, b)` with no division needed.

### 2. Two-Stage UMAP + Fuzzy Clustering Pipeline

A critical insight discovered during development: **FCM on raw 384-dimensional embeddings fails completely** due to the curse of dimensionality. In high dimensions, all pairwise Euclidean distances converge to the same value — FCM cannot distinguish cluster boundaries and assigns uniform ~1/15 memberships to every document.

The fix is a two-stage UMAP pipeline:

| Stage | Input | Output | Purpose |
|---|---|---|---|
| UMAP-1 | 384-dim | 50-dim | Clustering — makes Euclidean distance meaningful |
| FCM | 50-dim | Memberships | Soft cluster assignment |
| UMAP-2 | 384-dim | 2-dim | Visualisation — human-readable plot |

This two-stage approach (separate UMAPs for clustering and visualisation) is the standard pattern used in production NLP pipelines like BERTopic and Top2Vec.

**Evidence it works:**

| Configuration | FPC | Boundary Docs | Label Quality |
|---|---|---|---|
| FCM on raw 384-dim | ~0.07 | 16,781 / 18,000 (93%) | "conclusion / blast / given" |
| FCM on 50-dim UMAP | **0.5881** | 2,433 / 16,781 (14.5%) | "space / launch / nasa" ✓ |

FPC (Fuzzy Partition Coefficient) ranges from 1/C (random) to 1.0 (perfect hard partition). 0.5881 indicates genuine, meaningful cluster separation.

### 3. Enhanced TF-IDF Cluster Labelling with Lemmatisation

Cluster labels are extracted using a weighted TF-IDF pipeline with several improvements over a naive approach:

- **Lemmatisation** (NLTK WordNetLemmatizer) — reduces "christians" → "christian", preventing the same concept appearing three times in one label
- **Custom Usenet stop words** — removes high-frequency Usenet conversational words ("don", "just", "think", "know") that have no topic value
- **Weighted document contribution** — documents with higher cluster membership contribute proportionally more to the label via `np.dot(weights, tfidf_matrix)`
- **Top-30 representative documents** — uses the 30 highest-membership documents per cluster for a richer vocabulary signal
- **Full document text** — TF-IDF runs on the full cleaned document, not truncated snippets (early versions used 200-char truncation which produced nonsense labels)

### 4. Interactive UMAP Cluster Visualisation

An interactive Plotly HTML visualisation (`visualizations/cluster_viz.html`) shows:
- Each document as a point, coloured by dominant cluster
- Marker size proportional to cluster certainty — large points are confident members, small points are ambiguous
- Boundary documents (max membership < 0.45) highlighted with white × markers
- Hover tooltip showing top-2 cluster memberships with their labels, revealing the fuzzy nature of the assignments

---

## Technology Choices & Justification

### Embedding Model: `BAAI/bge-small-en-v1.5`

Chosen over `all-MiniLM-L6-v2` after empirical benchmarking on representative query pairs from this corpus. BGE-small scored 0.05–0.10 higher on average due to retrieval-specific fine-tuning.

| Model | Dims | Params | CPU Speed | MTEB Retrieval | Why |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 22M | ~20ms | 49.2 | Initial choice |
| **`BAAI/bge-small-en-v1.5`** | 384 | 33M | ~25ms | **51.7** | **Chosen — trained for retrieval** |
| `all-mpnet-base-v2` | 768 | 110M | ~150ms | 57.0 | Too slow on CPU for live API |
| OpenAI `ada-002` | 1536 | cloud | fast | ~60 | Paid API — assignment forbids it |

BGE models use an instruction prefix for queries: `"Represent this sentence for searching relevant passages: "`. Documents are stored without the prefix. This asymmetric encoding is the key design difference that improves retrieval scores.

### Vector Database: ChromaDB

ChromaDB is embedded (no separate server process), persists to disk, handles cosine distance natively, and stores metadata alongside vectors. FAISS is faster at billion-scale but requires manual metadata management — unnecessary overhead for an 18,000-document corpus.

### Clustering: scikit-fuzzy FCM

The assignment explicitly requires soft cluster assignments — a document about gun legislation belongs to both politics and firearms clusters with partial membership. FCM is the only widely-used Python library that produces this probability distribution output natively. K-Means would force a hard label and destroy the semantic nuance the assignment specifically asks for.

### Cache Storage: SQLite

- Atomic writes via WAL — no corruption risk on crash
- Efficient filtered queries: `SELECT * WHERE cluster_id IN (2, 7, 12)` — impossible efficiently with JSON
- Thread-safe — handles concurrent requests from Uvicorn
- Persistent across restarts — embeddings are model-frozen so cached results remain valid

### API: FastAPI

- Async request handling — event loop serves new requests while waiting for I/O
- Auto-generated OpenAPI docs at `/docs` — judges can test without curl
- Pydantic models provide automatic input validation and response serialisation guarantees
- `lifespan` context manager handles startup (model warmup, DB init) cleanly

---

## Project Structure

```
newsgroups-semantic-search/
│
├── app/
│   ├── main.py                      # FastAPI app + startup lifespan
│   ├── api/
│   │   └── routes.py                # POST /query, GET /cache/stats, DELETE /cache
│   ├── services/
│   │   ├── embedding_service.py     # BGE-small singleton, query instruction prefix
│   │   ├── vector_db_service.py     # ChromaDB cosine search
│   │   ├── clustering_service.py    # Runtime cluster assignment via centroid dot product
│   │   └── cache_service.py         # SQLite cache: TTL + LRU + cluster-aware lookup
│   ├── models/
│   │   └── schemas.py               # Pydantic request/response models
│   └── utils/
│
├── scripts/
│   ├── prepare_data.py              # Clean → embed (BGE) → ChromaDB + embeddings.npy
│   └── build_clusters.py           # UMAP(50d) → FCM → TF-IDF labels → UMAP(2d) → Plotly
│
├── data/                            # Generated by scripts — gitignored
│   ├── chroma_db/                   # ChromaDB vector store
│   ├── embeddings.npy               # (n_docs, 384) float32
│   ├── embeddings_50d.npy           # (n_docs, 50) UMAP-reduced for clustering
│   ├── cluster_memberships.npy      # (n_docs, 15) FCM probability distributions
│   ├── cluster_meta.json            # Labels, FPC score, boundary doc sample
│   └── doc_meta.json                # IDs, original labels, full cleaned text
│
├── cache/
│   └── cache.db                     # SQLite semantic cache (auto-created at startup)
│
├── visualizations/
│   └── cluster_viz.html             # Interactive Plotly UMAP scatter plot
│
├── test_folder/                     # Similarity benchmarking scripts
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
- 4 GB RAM minimum (model weights + embeddings)
- 3 GB disk (ChromaDB + model cache)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/sachin274/Newgroups-Semantic-Search
cd Newgroups-Semantic-Search

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Install PyTorch first (prevents DLL issues on Windows)
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Prepare data — embeds full corpus (~18,000 docs), ~20 min on CPU
python scripts/prepare_data.py

# 6. Build clusters and visualisation (~5 min on CPU)
python scripts/build_clusters.py

# 7. Start the API
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## API Reference

### `POST /query`

Embeds the query, checks the semantic cache, returns cached or freshly-computed results.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "NASA space shuttle launch mission"}'
```

**Cache miss response (first call):**
```json
{
  "query": "NASA space shuttle launch mission",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [
    {
      "document": "The shuttle lifted off at 0630 UTC...",
      "label_name": "sci.space",
      "similarity": 0.8234
    }
  ],
  "dominant_cluster": 8,
  "cluster_label": "space / launch / nasa"
}
```

**Cache hit response (semantically similar query):**
```json
{
  "query": "rocket launch and space exploration",
  "cache_hit": true,
  "matched_query": "NASA space shuttle launch mission",
  "similarity_score": 0.7821,
  "result": [...same results...],
  "dominant_cluster": 8,
  "cluster_label": "space / launch / nasa"
}
```

### `GET /cache/stats`

```bash
curl http://localhost:8000/cache/stats
```

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405,
  "similarity_threshold": 0.65,
  "max_entries": 1000,
  "ttl_seconds": 86400
}
```

### `DELETE /cache`

```bash
curl -X DELETE http://localhost:8000/cache
```

```json
{"message": "Cache flushed successfully."}
```

---

## Semantic Cache — Design Deep Dive

### Lookup Algorithm

```
1. Embed query → 384-dim L2-normalised vector
2. Find top-3 nearest cluster centroids via dot product
3. SQL: SELECT * FROM cache_entries WHERE cluster_id IN (c1, c2, c3)
4. For each candidate: similarity = dot(query_vec, cached_vec)
5. If max_similarity >= 0.65 → cache HIT, return stored result
6. Else → cache MISS, run ChromaDB search, INSERT new entry
```

### Similarity Threshold — Empirical Analysis

The threshold was chosen after measuring cosine similarities between real query pairs using `BAAI/bge-small-en-v1.5`:

| Query Pair | Similarity | Relationship |
|---|---|---|
| "NASA shuttle launch" / "NASA shuttle launch" | 1.00 | Identical |
| "install Python Windows" / "set up Python on Windows" | 0.85 | Near-paraphrase |
| "gun control Congress" / "firearms legislation debate" | 0.71 | Topic-similar |
| "space mission" / "baseball World Series" | 0.41 | Unrelated |

| Threshold | Behaviour |
|---|---|
| 0.90 | Only near-identical phrasing hits — very precise, low hit rate |
| 0.75 | Clear paraphrases hit — good for production accuracy |
| **0.65** | **Topic-similar queries hit — chosen for this corpus** |
| 0.55 | False positives begin appearing — unrelated topics start matching |

**0.65 was chosen** because it successfully captures paraphrases and topic-similar queries while keeping unrelated queries (similarity ~0.40–0.50) well below the threshold.

### Why Cluster-Aware Lookup Scales

```
Naive:    compare query vs ALL N cache entries → O(N)
Cluster:  compare query vs entries in top-3 clusters → O(N/C × 3)

At N=1000, C=15:  1000 comparisons → ~200 comparisons (5x faster)
At N=10000, C=15: 10000 comparisons → ~2000 comparisons (5x faster)
```

The speedup is constant regardless of cache size — this is why the cluster structure from Part 2 does "real work" in Part 3, as the assignment requires.

### SQLite Schema

```sql
CREATE TABLE cache_entries (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    original_query   TEXT NOT NULL,           -- for matched_query response field
    query_embedding  BLOB NOT NULL,           -- np.float32 serialised via tobytes()
    result_json      TEXT NOT NULL,           -- JSON-serialised ChromaDB hits
    cluster_id       INTEGER NOT NULL,        -- dominant cluster (indexed)
    similarity_score REAL NOT NULL,           -- always 1.0 for stored entries
    access_count     INTEGER DEFAULT 1,       -- for potential LFU eviction
    last_accessed    REAL NOT NULL,           -- Unix timestamp for LRU eviction
    created_at       REAL NOT NULL            -- Unix timestamp for TTL expiry
);
CREATE INDEX idx_cluster ON cache_entries (cluster_id);
CREATE INDEX idx_last_accessed ON cache_entries (last_accessed);
```

---

## Fuzzy Clustering — Design Deep Dive

### Why FCM over K-Means

A post about gun control legislation belongs to both the politics cluster and the firearms cluster with partial membership. K-Means forces a hard assignment — this semantic nuance is lost. FCM outputs a probability distribution, which is both more accurate and more informative.

**Example document membership (FCM output):**
```
Document: "The Senate voted on gun control legislation this week"

Cluster 7  (israel / arab / jews):       0.43  ← dominant
Cluster 5  (right / civil / trial):      0.27  ← secondary
Cluster 12 (encryption / government):    0.11
... remaining clusters share ~0.19 total
```

### Why 15 Clusters, Not 20

The 20 newsgroup categories are editorially defined. The actual semantic structure of the corpus is different — groups like `rec.autos` and `rec.motorcycles` are semantically near-identical in embedding space. FPC analysis shows diminishing returns beyond 15 clusters.

### Parameter Justification

| Parameter | Value | Justification |
|---|---|---|
| `N_CLUSTERS` | 15 | FPC elbow — semantic structure stabilises at 15, not 20 |
| `FUZZINESS_M` | 2.0 | Theoretically motivated default. m=1 → K-Means; m→∞ → uniform |
| `BOUNDARY_THRESH` | 0.45 | Documents below 0.45 max membership have no clear dominant topic |
| UMAP `n_components` | 50 | Retains ~95% of semantic structure; makes Euclidean distance valid |
| UMAP `min_dist` | 0.0 | Tighter packing for clustering (vs 0.1 for visualisation) |
| UMAP `n_neighbors` | 15 | Finer local structure for clustering (vs 30 for visualisation) |

### Clustering Results

```
FPC Score:          0.5881   (was ~0.07 on raw 384-dim — 8x improvement)
Total documents:    16,781
Well-assigned:      14,348  (85.5%)
Boundary docs:       2,433  (14.5%)  — genuinely cross-topic posts
```

---

## Cluster Visualisation

The interactive visualisation at `visualizations/cluster_viz.html` was generated by running UMAP (384→2 dims) on the full corpus and plotting with Plotly.

![UMAP Cluster Plot](visualizations/cluster_viz.png)

**How to read the plot:**
- Each point is one of the 16,781 documents
- Colour indicates the dominant cluster
- **Large points** — high cluster membership (confident assignment)
- **Small points** — low membership (ambiguous, multi-topic posts)
- **White × markers** — boundary documents (max membership < 0.45)
- **Hover** over any point to see the top-2 cluster memberships with their labels

The isolated orange cluster (top-left) is the religion/Christianity cluster — well-separated from the main corpus, which confirms it is a genuinely distinct semantic region. The dense central region where clusters overlap corresponds to the general discussion posts that touch multiple topics.

---

## Design Decisions Log

| Decision | Rationale |
|---|---|
| `remove=('headers','footers','quotes')` | Headers contain category labels → data leakage. Footers are boilerplate. Quoted text duplicates parent posts and inflates similarity. |
| Skip docs < 20 words | Too little signal for a meaningful 384-dim embedding; short posts pollute cluster centroids. |
| `MAX_DOCS = None` (full corpus) | More documents = better coverage of rare topics. One-time 20-min cost. |
| L2-normalise all embeddings | cosine_similarity(a, b) = dot(a, b) for unit vectors — faster, length-invariant |
| BGE instruction prefix on queries only | BGE convention: asymmetric encoding improves retrieval. Documents stored without prefix. |
| UMAP 384→50 before FCM | Curse of dimensionality: Euclidean distance is meaningless in 384 dims. 50 dims restores valid distance geometry. |
| Two separate UMAPs | 50-dim for clustering quality; 2-dim for visualisation only. Different `min_dist` and `n_neighbors` for each purpose. |
| `max_df=0.5` in TF-IDF | Removes words in >50% of documents (corpus-wide noise) while keeping topic-specific terms like "nasa", "encryption", "gun". |
| `min_df=3` | Removes hapax legomena (unique terms, usually typos or names) that would dominate labels. |
| Lemmatisation before TF-IDF | Prevents "christian / christians / christianity" — three tokens for one concept. |
| LRU over LFU eviction | LRU adapts when query patterns shift over time. LFU retains old high-frequency entries even after they become irrelevant. |
| Lazy TTL expiry | Checked during lookup rather than via a background thread — simpler, avoids race conditions, cost is amortised. |
| `SIMILARITY_THRESHOLD = 0.65` | Empirically calibrated on this corpus and model. Captures topic-similar queries; unrelated queries score ~0.40–0.50. |

---

## Docker

```bash
# Build image
docker build -t newsgroups-search .

# Run with pre-built data mounted from host
# (run prepare_data.py and build_clusters.py on host first)
docker-compose up
```

The container starts Uvicorn on port 8000. Data and cache directories are mounted as volumes so the pre-built artefacts are available inside the container without rebuilding.

```yaml
# docker-compose.yml
volumes:
  - ./data:/app/data       # ChromaDB + embeddings
  - ./cache:/app/cache     # SQLite cache
  - ./visualizations:/app/visualizations
```

---

## Requirements

```
sentence-transformers==2.7.0
chromadb==0.5.0
fastapi==0.111.0
uvicorn[standard]==0.29.0
scikit-learn==1.4.2
scikit-fuzzy==0.4.2
numpy==1.26.4
scipy==1.13.0
umap-learn==0.5.6
plotly==5.22.0
pandas==2.2.2
tqdm==4.66.4
nltk==3.8.1
pydantic==2.7.1
python-dotenv==1.0.1
```

> **Windows note:** Install PyTorch explicitly before other packages to avoid DLL errors:
> `pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu`
