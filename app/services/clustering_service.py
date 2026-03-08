"""
app/services/clustering_service.py
====================================
Loads pre-computed cluster artefacts and provides:
1. Dominant cluster for a new query (via nearest-centroid lookup)
2. Cluster label strings for the API response

This module does NOT re-run FCM at runtime — that would be prohibitively
slow per-request. Instead, at query time we assign the query to its nearest
cluster centroid using cosine similarity, which is O(n_clusters).
"""

import os
import json
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
CLUSTER_META_PATH  = os.path.join(DATA_DIR, "cluster_meta.json")
MEMBERSHIPS_PATH   = os.path.join(DATA_DIR, "cluster_memberships.npy")

# Pre-computed cluster centroids are stored in the ChromaDB collection as
# full document embeddings. We approximate the centroid as the mean of
# the top-weighted documents per cluster and cache it here at startup.
_cluster_meta: dict | None = None
_cluster_centroids: np.ndarray | None = None


def _load_meta():
    global _cluster_meta
    if _cluster_meta is None:
        with open(CLUSTER_META_PATH) as f:
            _cluster_meta = json.load(f)
    return _cluster_meta


def _load_centroids():
    """
    Build approximate cluster centroids from the saved membership matrix
    and pre-computed embeddings at startup (once).
    """
    global _cluster_centroids
    if _cluster_centroids is None:
        emb_path = os.path.join(DATA_DIR, "embeddings.npy")
        memberships = np.load(MEMBERSHIPS_PATH)  # (n_docs, n_clusters)
        embeddings  = np.load(emb_path)          # (n_docs, 384)

        # Fuzzy centroid: weighted mean of all embeddings for each cluster
        n_clusters = memberships.shape[1]
        centroids = np.zeros((n_clusters, embeddings.shape[1]))
        for c in range(n_clusters):
            w = memberships[:, c]
            centroids[c] = (embeddings * w[:, None]).sum(axis=0) / (w.sum() + 1e-9)

        # L2-normalise so we can use dot product as cosine similarity
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        _cluster_centroids = centroids / (norms + 1e-9)
    return _cluster_centroids


def get_dominant_cluster(query_embedding: np.ndarray) -> int:
    """
    Assign a query to its nearest cluster centroid via cosine similarity.
    Returns the cluster index (0-based).
    """
    centroids = _load_centroids()
    # query_embedding is already L2-normalised
    sims = centroids @ query_embedding   # shape (n_clusters,)
    return int(np.argmax(sims))


def get_cluster_label(cluster_id: int) -> str:
    meta = _load_meta()
    return meta["cluster_labels"].get(str(cluster_id), f"Cluster {cluster_id}")


def get_cluster_ids_sorted_by_similarity(
        query_embedding: np.ndarray, top_k: int = 3) -> list[int]:
    """
    Return the top_k cluster IDs most similar to the query embedding.
    Used by the semantic cache to narrow lookup scope when the cache is large.
    """
    centroids = _load_centroids()
    sims = centroids @ query_embedding
    sorted_ids = np.argsort(sims)[::-1][:top_k]
    return sorted_ids.tolist()
