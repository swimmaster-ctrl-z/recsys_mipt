from __future__ import annotations

import numpy as np

from .common import item_gram

def cosine_from_gram(gram: np.ndarray, shrinkage: float) -> np.ndarray:
    norms = np.sqrt(np.diag(gram))
    denominator = norms[:, None] * norms[None, :]
    similarity = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 0)
    if shrinkage:
        similarity *= gram / (gram + shrinkage)
    np.fill_diagonal(similarity, 0.0)
    return similarity


class ItemKNN:
    def __init__(self, neighbors: int = 100, shrinkage: float = 10.0):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, dataset: Dataset) -> "ItemKNN":
        self.dataset = dataset
        similarity = cosine_from_gram(item_gram(dataset), self.shrinkage)
        if self.neighbors < dataset.n_items:
            for row in similarity:
                discard = np.argsort(row, kind="stable")[: -self.neighbors]
                row[discard] = 0.0
        self.similarity = similarity
        self.seen = dataset.train.groupby("u")["i"].agg(list).to_dict()
        return self

    def score(self, user: int) -> np.ndarray:
        items = self.seen.get(user, [])
        if not items:
            return np.zeros(self.dataset.n_items)
        return self.similarity[items].sum(axis=0)
