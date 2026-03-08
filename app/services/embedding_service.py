from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models use an instruction prefix for queries only.
# This tells the model "this is a search query, find relevant passages"
# Documents stored in ChromaDB do NOT get this prefix.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_query(text: str) -> np.ndarray:
    model = get_model()
    # Prefix only on the query side, not on stored documents
    vec = model.encode(
        [QUERY_INSTRUCTION + text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return vec[0]

def embed_documents(texts: list) -> np.ndarray:
    model = get_model()
    # No prefix for documents — BGE convention
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=True
    )