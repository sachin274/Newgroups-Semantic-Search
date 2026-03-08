"""
app/services/embedding_service.py
==================================
Singleton wrapper around SentenceTransformer.

We load the model once at startup and reuse it for every request.
Loading a 22M-parameter model takes ~1–2 seconds; doing it per-request
would make the API unusably slow.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singleton — loaded once when the module is first imported
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string and return a normalised 384-dim vector.

    Normalisation ensures that cosine_similarity(a, b) == dot(a, b),
    which is faster and avoids a division operation in the cache lookup.
    """
    model = get_model()
    vec = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return vec[0]   # shape (384,)
