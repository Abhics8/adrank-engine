"""ANN index over ad embeddings. Uses FAISS when available; falls back to
exact brute-force search (sklearn) otherwise, so the pipeline still runs on
machines where installing faiss is impractical -- the recall@k eval reports
which backend produced the number."""
from __future__ import annotations

import numpy as np

try:
    import faiss

    _HAS_FAISS = True
except ImportError:  # pragma: no cover - exercised only on faiss-less envs
    from sklearn.neighbors import NearestNeighbors

    _HAS_FAISS = False


class AdIndex:
    def __init__(self, ad_embeddings: np.ndarray, ad_ids: np.ndarray):
        self.ad_ids = ad_ids
        self.backend = "faiss" if _HAS_FAISS else "sklearn-bruteforce"
        if _HAS_FAISS:
            self.index = faiss.IndexFlatIP(ad_embeddings.shape[1])
            self.index.add(ad_embeddings.astype(np.float32))
        else:
            self.index = NearestNeighbors(metric="cosine").fit(ad_embeddings)

    def search(self, user_embeddings: np.ndarray, k: int) -> np.ndarray:
        """Returns array of shape (n_users, k) of ad_ids, best match first."""
        if _HAS_FAISS:
            _, idx = self.index.search(user_embeddings.astype(np.float32), k)
        else:
            _, idx = self.index.kneighbors(user_embeddings, n_neighbors=k)
        return self.ad_ids[idx]
