"""
scripts/prepare_data.py
=======================
Loads the 20 Newsgroups dataset, cleans it, generates sentence-transformer
embeddings, and persists everything to ChromaDB.

Design decisions (justified here per assignment brief):
- We use sklearn's fetch_20newsgroups with remove=('headers','footers','quotes')
  because headers contain the true category label (data leakage), footers are
  boilerplate, and quoted text duplicates parent-post content.
- We cap documents at MAX_DOCS to keep embedding time reasonable on CPU;
  the full 18,846 docs can be enabled by setting MAX_DOCS = None.
- Embedding model: all-MiniLM-L6-v2 (22M params, 384-dim). Fast on CPU,
  strong semantic signal, MIT license. Better than TF-IDF because it captures
  meaning not just token overlap; smaller than all-mpnet-base-v2 so it runs
  without a GPU.
- ChromaDB: embedded (no server), stores vectors + metadata on disk, supports
  cosine similarity out of the box. FAISS is faster at scale but requires more
  manual plumbing for metadata filtering — unnecessary here.
"""

import os
import re
import json
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ── Config ────────────────────────────────────────────────────────────────────
MAX_DOCS = 5000          # Set None to embed the full corpus (slow on CPU)
BATCH_SIZE = 64
MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
META_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "doc_meta.json")
EMB_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.npy")
# ─────────────────────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """
    Light cleaning pipeline for newsgroup posts.

    Steps:
    1. Strip email addresses — they carry no semantic content and would bias
       embeddings towards specific senders.
    2. Remove URLs — mostly dead links in a 30-year-old corpus.
    3. Collapse whitespace — the model tokeniser handles this anyway, but
       keeping clean text makes debugging easier.
    4. Truncate to 512 words — MiniLM has a 256-token context window; feeding
       more tokens just gets silently truncated by the tokeniser. Truncating
       early keeps memory predictable.

    We deliberately do NOT stem, lemmatise, or lower-case. Sentence-transformers
    are trained on raw cased text and perform better with it.
    """
    text = re.sub(r'\S+@\S+', '', text)                  # emails
    text = re.sub(r'http\S+|www\.\S+', '', text)         # URLs
    text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)         # special chars
    text = re.sub(r'\s+', ' ', text).strip()             # whitespace
    words = text.split()
    return ' '.join(words[:512])


def load_and_clean():
    print("Loading 20 Newsgroups dataset …")
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
        # Skip documents shorter than 20 words — they carry too little signal
        # to produce a meaningful embedding and would pollute cluster centroids.
        if len(cleaned.split()) < 20:
            continue
        docs.append(cleaned)
        labels.append(int(target))
        label_names.append(data.target_names[target])
        doc_ids.append(f"doc_{i}")

    print(f"Kept {len(docs)} documents after cleaning.")
    return docs, labels, label_names, doc_ids


def embed_documents(docs):
    print(f"Loading embedding model: {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings (this may take a few minutes on CPU) …")
    embeddings = model.encode(
        docs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalise so cosine sim == dot product
        convert_to_numpy=True
    )
    return embeddings


def store_in_chromadb(docs, embeddings, labels, label_names, doc_ids):
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if re-running
    try:
        client.delete_collection("newsgroups")
    except Exception:
        pass

    collection = client.create_collection(
        name="newsgroups",
        # We store pre-normalised embeddings and use cosine distance.
        # Cosine is preferred over L2 for text because document length should
        # not affect similarity — two short posts on the same topic should be
        # as close as two long ones.
        metadata={"hnsw:space": "cosine"}
    )

    print("Storing documents in ChromaDB …")
    for batch_start in tqdm(range(0, len(docs), BATCH_SIZE)):
        batch_end = batch_start + BATCH_SIZE
        collection.add(
            ids=doc_ids[batch_start:batch_end],
            embeddings=embeddings[batch_start:batch_end].tolist(),
            documents=docs[batch_start:batch_end],
            metadatas=[
                {"label": labels[i], "label_name": label_names[i]}
                for i in range(batch_start, min(batch_end, len(docs)))
            ]
        )

    print(f"Stored {collection.count()} documents in ChromaDB.")
    return collection


def main():
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)

    docs, labels, label_names, doc_ids = load_and_clean()
    embeddings = embed_documents(docs)

    # Persist embeddings as numpy array for clustering script
    np.save(EMB_PATH, embeddings)
    print(f"Embeddings saved to {EMB_PATH}  shape={embeddings.shape}")

    # Persist metadata for clustering script
    # We save the full cleaned text (not truncated to 200 chars) so that
    # TF-IDF in build_clusters.py sees the complete document vocabulary.
    # Truncating to 200 chars caused cluster labels like "conclusion/blast/given"
    # because the snippet was too short to contain topic-representative terms.
    meta = {
        "doc_ids": doc_ids,
        "labels": labels,
        "label_names": label_names,
        "docs_preview": docs   # full cleaned text for TF-IDF label extraction
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f)
    print(f"Metadata saved to {META_PATH}")

    store_in_chromadb(docs, embeddings, labels, label_names, doc_ids)
    print("\n✓ Data preparation complete. Run build_clusters.py next.")


if __name__ == "__main__":
    main()




