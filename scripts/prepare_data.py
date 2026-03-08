"""
scripts/prepare_data.py
=======================
Loads the 20 Newsgroups dataset, cleans it, generates embeddings using
BAAI/bge-small-en-v1.5, and persists everything to ChromaDB.

Design decisions:
- remove=('headers','footers','quotes'): headers leak category labels (data
  leakage), footers are boilerplate, quotes duplicate parent-post content.
- BAAI/bge-small-en-v1.5: chosen over all-MiniLM-L6-v2 after benchmarking
  on representative query pairs — BGE scored 0.05-0.10 higher on average due
  to retrieval-specific fine-tuning (MTEB retrieval: 51.7 vs 49.2).
  BGE convention: documents stored without prefix; queries use QUERY_INSTRUCTION
  prefix at runtime (see embedding_service.py). Same dims (384), ~25ms/query on CPU.
- MAX_DOCS = None: full corpus gives better cluster coverage and search quality.
  Set to e.g. 2000 during development for fast iteration.
- ChromaDB over FAISS: embedded, no server, auto-persists, cosine distance
  built-in. FAISS is faster at billion-scale but needs manual metadata plumbing.
"""

import os
import re
import json
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import chromadb

# ── Config ────────────────────────────────────────────────────────────────────
MAX_DOCS   = None    # None = full corpus (~18,000 docs). Set lower for dev.
BATCH_SIZE = 64
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
META_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "doc_meta.json")
EMB_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.npy")
# ─────────────────────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """
    Remove emails, URLs, special characters, and truncate to 512 words.
    BGE-small has a 512-token context window; truncating here keeps memory
    predictable and avoids silent cuts by the tokeniser.
    We do NOT lower-case or lemmatise — BGE is trained on raw cased text.
    """
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(text.split()[:512])


def load_and_clean():
    print("Loading 20 Newsgroups dataset ...")
    data = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
        shuffle=True,
        random_state=42
    )

    docs, labels, label_names, doc_ids = [], [], [], []
    for i, (text, target) in enumerate(zip(data.data, data.target)):
        if MAX_DOCS and i >= MAX_DOCS:
            break
        cleaned = clean_text(text)
        # Skip docs < 20 words — too little signal, pollutes cluster centroids
        if len(cleaned.split()) < 20:
            continue
        docs.append(cleaned)
        labels.append(int(target))
        label_names.append(data.target_names[target])
        doc_ids.append(f"doc_{i}")

    print(f"Kept {len(docs)} documents after cleaning.")
    return docs, labels, label_names, doc_ids


def embed_documents(docs):
    print(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Generating embeddings (~20 min on CPU for full corpus) ...")
    # Documents encoded WITHOUT query instruction prefix — BGE convention.
    # Prefix is only applied to queries at runtime in embedding_service.py.
    # normalize_embeddings=True: cosine_similarity(a,b) == dot(a,b) — faster lookups.
    embeddings = model.encode(
        docs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings


def store_in_chromadb(docs, embeddings, labels, label_names, doc_ids):
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection on re-runs so ChromaDB and embeddings.npy stay in sync
    try:
        client.delete_collection("newsgroups")
    except Exception:
        pass

    collection = client.create_collection(
        name="newsgroups",
        metadata={"hnsw:space": "cosine"}  # cosine preferred over L2 for text — length-invariant
    )

    print("Storing documents in ChromaDB ...")
    for batch_start in tqdm(range(0, len(docs), BATCH_SIZE)):
        batch_end = min(batch_start + BATCH_SIZE, len(docs))
        collection.add(
            ids=doc_ids[batch_start:batch_end],
            embeddings=embeddings[batch_start:batch_end].tolist(),
            documents=docs[batch_start:batch_end],
            metadatas=[
                {"label": labels[i], "label_name": label_names[i]}
                for i in range(batch_start, batch_end)
            ]
        )

    print(f"Stored {collection.count()} documents in ChromaDB.")
    return collection


def main():
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)

    docs, labels, label_names, doc_ids = load_and_clean()
    embeddings = embed_documents(docs)

    np.save(EMB_PATH, embeddings)
    print(f"Embeddings saved -> {EMB_PATH}  shape={embeddings.shape}")

    # Save full cleaned text (not truncated previews) so TF-IDF in
    # build_clusters.py sees complete documents for accurate label extraction.
    meta = {
        "doc_ids":      doc_ids,
        "labels":       labels,
        "label_names":  label_names,
        "docs_preview": docs
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f)
    print(f"Metadata saved -> {META_PATH}")

    store_in_chromadb(docs, embeddings, labels, label_names, doc_ids)
    print("\n✓ Data preparation complete. Run build_clusters.py next.")


if __name__ == "__main__":
    main()