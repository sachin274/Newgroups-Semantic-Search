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
        max_df=0.5,               # ignore terms that appear in >70% of docs
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