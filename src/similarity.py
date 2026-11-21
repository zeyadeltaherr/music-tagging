"""Cosine-similarity search over track embeddings (written from scratch, NumPy only)."""
import numpy as np


class SimilarityIndex:
    def __init__(self):
        self.emb = None
        self.ids = None

    @staticmethod
    def _normalize(x):
        x = np.asarray(x, dtype=np.float32)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def fit(self, embeddings, ids=None):
        self.emb = self._normalize(embeddings)
        self.ids = np.arange(len(self.emb)) if ids is None else np.asarray(ids)
        return self

    def search(self, queries, k=5, exclude_self=False, chunk=1024):
        """Return (indices, scores), each (n_queries, k), sorted by similarity (desc).
        exclude_self=True assumes queries are the indexed items in the same order."""
        q_all = self._normalize(queries)
        k = min(k, len(self.emb) - (1 if exclude_self else 0))
        all_idx, all_sc = [], []
        for start in range(0, len(q_all), chunk):
            q = q_all[start:start + chunk]
            sims = q @ self.emb.T
            if exclude_self:
                rows = np.arange(len(q))
                sims[rows, start + rows] = -np.inf
            part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            part_sc = np.take_along_axis(sims, part, axis=1)
            order = np.argsort(-part_sc, axis=1)
            all_idx.append(np.take_along_axis(part, order, axis=1))
            all_sc.append(np.take_along_axis(part_sc, order, axis=1))
        return np.concatenate(all_idx), np.concatenate(all_sc)

    def save(self, path):
        np.savez(path, emb=self.emb, ids=self.ids)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        obj = cls()
        obj.emb, obj.ids = d["emb"], d["ids"]
        return obj
