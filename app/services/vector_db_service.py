"""
app/services/vector_db_service.py
===================================
Thin wrapper around ChromaDB for semantic document retrieval.

On first import the collection is opened (not rebuilt). The heavy
data-preparation work (embedding + storing) is done by scripts/prepare_data.py.
"""

import os
import chromadb

CHROMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "chroma_db"
)

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_collection("newsgroups")
    return _collection


def semantic_search(query_embedding, n_results: int = 5) -> list[dict]:
    """
    Return the top-n_results most semantically similar documents to the
    query embedding.

    ChromaDB returns results sorted by distance (cosine distance = 1 - sim).
    We convert back to similarity for the API response.
    """
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        hits.append({
            "document": doc[:500],    # truncate for API response readability
            "label_name": meta.get("label_name", "unknown"),
            "similarity": round(1.0 - dist, 4)
        })
    return hits
