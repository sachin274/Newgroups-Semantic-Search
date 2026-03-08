"""
scripts/build_clusters.py
=========================
Performs Fuzzy C-Means clustering on the pre-computed embeddings, generates
UMAP visualisations, extracts cluster labels via TF-IDF, and persists cluster
membership probabilities for use by the cache and API.

Key design decisions:
- Fuzzy C-Means (FCM) over K-Means: a document about "gun legislation" should
  belong to both politics AND firearms clusters with partial membership. Hard
  assignment destroys this nuance. FCM outputs a probability distribution per
  document, which is exactly what the assignment requires.
- Number of clusters: we choose 15, not 20. The 20 newsgroup labels are
  editorially defined; the *semantic* structure of the corpus is different.
  Groups like rec.autos and rec.motorcycles are semantically near-identical.
  We use the Fuzzy Partition Coefficient (FPC) curve to justify 15 — see the
  elbow analysis below.
- UMAP over t-SNE: UMAP preserves global structure better, is faster, and
  produces cleaner cluster boundaries for 384-dim vectors. t-SNE is
  non-deterministic and slow for 5000+ points.
- Fuzziness m=2.0: the standard default. m→1 approaches hard assignment;
  m→∞ makes all memberships equal. m=2 gives a good balance.
"""

import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import skfuzzy as fuzz
import plotly.graph_objects as go
import plotly.express as px
import umap

# ── Config ────────────────────────────────────────────────────────────────────
N_CLUSTERS    = 15       # Justified by FPC elbow (see find_optimal_clusters)
FUZZINESS_M   = 2.0      # Standard FCM fuzziness parameter
FCM_MAX_ITER  = 150
FCM_ERROR     = 0.005
BOUNDARY_THRESH = 0.15   # Max-membership below this → boundary document
UMAP_N_COMPONENTS = 2

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
VIZ_DIR   = os.path.join(os.path.dirname(__file__), "..", "visualizations")
EMB_PATH  = os.path.join(DATA_DIR, "embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "doc_meta.json")
CLUSTER_PATH = os.path.join(DATA_DIR, "cluster_memberships.npy")  # shape (n_docs, n_clusters)
CLUSTER_META_PATH = os.path.join(DATA_DIR, "cluster_meta.json")
# ─────────────────────────────────────────────────────────────────────────────


def find_optimal_clusters(embeddings_T, cluster_range=range(8, 22)):
    """
    Sweep over candidate cluster counts and record the Fuzzy Partition
    Coefficient (FPC). FPC ranges from 1/C (total fuzziness) to 1 (hard
    partition). We look for the elbow — adding more clusters yields diminishing
    returns in partition quality.

    This is NOT run every time (it's slow). Run it once manually to justify
    the N_CLUSTERS constant above. Results saved to data/fpc_curve.json.
    """
    print("Running FPC sweep to find optimal cluster count …")
    fpc_scores = {}
    for c in tqdm(cluster_range):
        _, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            embeddings_T, c, FUZZINESS_M,
            error=FCM_ERROR, maxiter=FCM_MAX_ITER, init=None
        )
        fpc_scores[c] = fpc
        print(f"  C={c}  FPC={fpc:.4f}")

    with open(os.path.join(DATA_DIR, "fpc_curve.json"), 'w') as f:
        json.dump(fpc_scores, f, indent=2)
    print("FPC curve saved.")
    return fpc_scores


def run_fuzzy_clustering(embeddings):
    """
    Run FCM. scikit-fuzzy expects data as (n_features, n_samples) — note the
    transpose relative to sklearn convention.

    Returns:
        cntr   : cluster centres  (n_clusters, n_features)
        u      : membership matrix (n_clusters, n_samples)  — probabilities
    """
    print(f"Running Fuzzy C-Means  C={N_CLUSTERS}  m={FUZZINESS_M} …")
    embeddings_T = embeddings.T   # (384, n_docs)

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        embeddings_T,
        N_CLUSTERS,
        FUZZINESS_M,
        error=FCM_ERROR,
        maxiter=FCM_MAX_ITER,
        init=None,
        seed=42
    )
    print(f"  Converged  FPC={fpc:.4f}")
    # u.T → (n_docs, n_clusters)  — each row sums to 1.0
    return cntr, u.T, fpc


def extract_cluster_labels(docs_preview, memberships):
    """
    For each cluster, collect the top-30 documents (by membership weight),
    extract TF-IDF keywords, and form a 3-word label.

    Key improvements over naive approach:
    - Use top-30 docs per cluster (not 10) for richer vocabulary signal
    - Filter out single-character tokens and purely numeric terms
    - Use only unigrams (ngram_range=(1,1)) — bigrams often produce
      unintelligible labels from short noisy newsgroup posts
    - Fit TF-IDF on ALL docs so IDF penalises corpus-wide common words,
      then score each cluster's representative docs against that IDF
    """
    print("Extracting TF-IDF cluster labels …")
    n_clusters = memberships.shape[1]
    labels = {}

    # Fit on full corpus so IDF correctly penalises common words
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 1),       # unigrams only — more reliable for short docs
        min_df=3,                 # ignore very rare terms
        max_df=0.7,               # ignore terms that appear in >70% of docs
        token_pattern=r'[a-zA-Z]{3,}'  # only alphabetic tokens ≥ 3 chars
    )
    vectorizer.fit(docs_preview)
    terms = vectorizer.get_feature_names_out()

    for c in range(n_clusters):
        weights = memberships[:, c]
        # Take top-30 documents by cluster membership weight
        top_idx = np.argsort(weights)[-30:]
        cluster_docs = [docs_preview[i] for i in top_idx]
        doc_weights = weights[top_idx]

        tfidf_matrix = vectorizer.transform(cluster_docs)

        # Weighted sum: documents with higher cluster membership contribute more
        # to the final term scores, so the label reflects the cluster core
        weighted_tfidf = np.asarray(tfidf_matrix.T.dot(doc_weights)).flatten()

        top_term_idx = np.argsort(weighted_tfidf)[-3:][::-1]
        top_terms = [terms[i] for i in top_term_idx]
        labels[c] = " / ".join(top_terms)
        print(f"  Cluster {c:2d}: {labels[c]}")

    return labels


def build_umap_projection(embeddings):
    """
    Reduce 384-dim embeddings to 2D for visualisation.

    UMAP parameters:
    - n_neighbors=30: balances local vs global structure. Smaller → more
      local detail; larger → smoother global topology.
    - min_dist=0.1: controls how tightly points are packed. 0.1 is a good
      default for showing cluster separation.
    - metric='cosine': consistent with our similarity metric elsewhere.
    """
    print("Running UMAP dimensionality reduction …")
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=30,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        verbose=False
    )
    projection = reducer.fit_transform(embeddings)
    return projection


def build_interactive_visualization(projection, memberships, label_names_orig,
                                    cluster_labels, doc_ids):
    """
    Build an interactive Plotly scatter plot showing:
    1. Dominant cluster colour per document
    2. Marker size ∝ certainty (large = confident, small = ambiguous)
    3. Hover showing top-2 cluster memberships → reveals fuzzy boundaries
    """
    print("Building interactive cluster visualisation …")
    n_docs = len(doc_ids)
    dominant = np.argmax(memberships, axis=1)
    max_membership = np.max(memberships, axis=1)

    # Identify boundary documents
    is_boundary = max_membership < BOUNDARY_THRESH

    hover_texts = []
    for i in range(n_docs):
        top2 = np.argsort(memberships[i])[-2:][::-1]
        parts = [f"C{c}: {memberships[i,c]:.2f} ({cluster_labels.get(c,'?')})"
                 for c in top2]
        boundary_tag = " ⚠ BOUNDARY" if is_boundary[i] else ""
        hover_texts.append(
            f"{doc_ids[i]}{boundary_tag}<br>" + "<br>".join(parts)
        )

    colors = px.colors.qualitative.Alphabet[:N_CLUSTERS]
    marker_colors = [colors[d % len(colors)] for d in dominant]

    fig = go.Figure()

    # Main scatter
    fig.add_trace(go.Scatter(
        x=projection[:, 0],
        y=projection[:, 1],
        mode='markers',
        marker=dict(
            color=marker_colors,
            size=6 + 10 * max_membership,   # size ∝ certainty
            opacity=0.7,
            line=dict(width=0)
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        name='Documents'
    ))

    # Highlight boundary documents
    boundary_idx = np.where(is_boundary)[0]
    if len(boundary_idx) > 0:
        fig.add_trace(go.Scatter(
            x=projection[boundary_idx, 0],
            y=projection[boundary_idx, 1],
            mode='markers',
            marker=dict(
                color='white',
                size=3,
                symbol='x',
                opacity=0.4
            ),
            text=[hover_texts[i] for i in boundary_idx],
            hovertemplate='%{text}<extra></extra>',
            name='Boundary docs'
        ))

    fig.update_layout(
        title="20 Newsgroups — Fuzzy Cluster Visualisation (UMAP 2D)",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        legend_title="Legend",
        width=1100,
        height=750,
        template="plotly_dark",
        hoverlabel=dict(font_size=11)
    )

    out_path = os.path.join(VIZ_DIR, "cluster_viz.html")
    fig.write_html(out_path)
    print(f"  Visualisation saved → {out_path}")
    return out_path


def main():
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Load pre-computed artefacts
    print("Loading embeddings and metadata …")
    embeddings = np.load(EMB_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)

    doc_ids      = meta["doc_ids"]
    label_names  = meta["label_names"]
    docs_preview = meta["docs_preview"]

    # ── Optional: run elbow analysis once to justify N_CLUSTERS ──────────────
    # find_optimal_clusters(embeddings.T)

    # ── Fuzzy Clustering ──────────────────────────────────────────────────────
    cntr, memberships, fpc = run_fuzzy_clustering(embeddings)

    # Save membership matrix — shape (n_docs, n_clusters)
    np.save(CLUSTER_PATH, memberships)
    print(f"  Memberships saved  shape={memberships.shape}")

    # ── Cluster Labels ────────────────────────────────────────────────────────
    cluster_labels = extract_cluster_labels(docs_preview, memberships)

    # ── Boundary Documents ────────────────────────────────────────────────────
    max_mem = np.max(memberships, axis=1)
    dominant = np.argmax(memberships, axis=1)
    boundary_mask = max_mem < BOUNDARY_THRESH
    boundary_docs = [
        {
            "doc_id": doc_ids[i],
            "max_membership": float(max_mem[i]),
            "memberships": {str(c): float(memberships[i, c]) for c in range(N_CLUSTERS)}
        }
        for i in np.where(boundary_mask)[0]
    ]
    print(f"  Boundary documents (max_mem < {BOUNDARY_THRESH}): {len(boundary_docs)}")

    # ── Persist cluster metadata ──────────────────────────────────────────────
    cluster_meta = {
        "n_clusters": N_CLUSTERS,
        "fuzziness_m": FUZZINESS_M,
        "fpc": float(fpc),
        "cluster_labels": cluster_labels,
        "boundary_threshold": BOUNDARY_THRESH,
        "n_boundary_docs": len(boundary_docs),
        "doc_dominant_cluster": {doc_ids[i]: int(dominant[i]) for i in range(len(doc_ids))},
        "boundary_docs_sample": boundary_docs[:20]  # first 20 for the API
    }
    with open(CLUSTER_META_PATH, 'w') as f:
        json.dump(cluster_meta, f, indent=2)
    print(f"  Cluster meta saved → {CLUSTER_META_PATH}")

    # ── UMAP + Visualisation ──────────────────────────────────────────────────
    projection = build_umap_projection(embeddings)
    np.save(os.path.join(DATA_DIR, "umap_projection.npy"), projection)
    build_interactive_visualization(
        projection, memberships, label_names, cluster_labels, doc_ids
    )

    print("\n✓ Clustering complete. Start the API with: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()