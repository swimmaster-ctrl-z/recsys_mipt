from __future__ import annotations

import numpy as np

from .common import item_gram

def coefficients_from_gram(gram: np.ndarray, regularization: float) -> np.ndarray:
    n_items = len(gram)
    precision = np.linalg.solve(gram + regularization * np.eye(n_items), np.eye(n_items))
    coefficients = -precision / np.diag(precision)[None, :]
    np.fill_diagonal(coefficients, 0.0)
    return coefficients


class EASE:
    def __init__(self, regularization: float = 300.0):
        self.regularization = regularization

    def fit(self, dataset: Dataset) -> "EASE":
        self.dataset = dataset
        self.coefficients = coefficients_from_gram(item_gram(dataset), self.regularization)
        self.seen = dataset.train.groupby("u")["i"].agg(list).to_dict()
        return self

    def score(self, user: int) -> np.ndarray:
        items = self.seen.get(user, [])
        if not items:
            return np.zeros(self.dataset.n_items)
        return self.coefficients[items].sum(axis=0)
