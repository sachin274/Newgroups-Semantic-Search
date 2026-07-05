# Newsgroups Semantic Search

A semantic search engine over the 20 Newsgroups corpus (~18,000 documents). Type a natural language query and get back the most semantically similar documents.

Built with:
- **BAAI/bge-small-en-v1.5** — sentence embeddings for semantic similarity
- **ChromaDB** — vector database with HNSW indexing for fast document retrieval
- **Fuzzy C-Means + two-stage UMAP** — soft topic clustering across the corpus
- **In-memory semantic cache** — returns results instantly for similar past queries without hitting the vector DB
- **FastAPI** — REST API with a benchmark endpoint to compare cache vs vector DB latency

---

## Architecture

```
Client → FastAPI → EmbeddingService (BGE)
                 → ClusteringService
                 → CacheService (Python dict)  ──→ return if cache hit
                 → VectorDBService (ChromaDB)  ──→ on cache miss
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Search — returns top documents + `retrieval_time_ms` |
| `POST` | `/benchmark` | Runs both cache and ChromaDB, returns both timings |
| `GET` | `/cache/stats` | Hit rate, entry count, threshold |
| `DELETE` | `/cache` | Flush the cache |

### Example

```bash
POST /query
{ "query": "nasa space shuttle launch" }
```

```json
{
  "cache_hit": false,
  "result": [{ "document": "...", "label_name": "sci.space", "similarity": 0.89 }],
  "cluster_label": "space/launch/nasa",
  "retrieval_time_ms": 9.0
}
```

---

## How to Run

```bash
# Install
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Build data — run once (~25 min)
python scripts/prepare_data.py
python scripts/build_clusters.py

# Start server
uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

Or with Docker:
```bash
docker-compose up
```

---

## Tech Stack

`Python` `FastAPI` `ChromaDB` `sentence-transformers` `scikit-fuzzy` `umap-learn` `Plotly`
