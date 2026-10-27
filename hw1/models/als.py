from __future__ import annotations

import numpy as np
import pandas as pd

def _group_rows(indices: np.ndarray, size: int) -> list[np.ndarray]:
    order = np.argsort(indices, kind="stable")
    counts = np.bincount(indices, minlength=size)
    return np.split(order, np.cumsum(counts)[:-1])


class ExplicitALS:
    def __init__(
        self,
        factors: int = 16,
        reg: float = 0.5,
        iterations: int = 5,
        seed: int = 42,
    ):
        self.factors = factors
        self.reg = reg
        self.iterations = iterations
        self.seed = seed

    def _target_values(self, train: pd.DataFrame) -> np.ndarray:
        return train.rating.to_numpy(dtype=float) / 5.0

    def fit(self, dataset: Dataset) -> "ExplicitALS":
        self.dataset = dataset
        train = dataset.train
        u = train.u.to_numpy(dtype=int)
        i = train.i.to_numpy(dtype=int)
        target = self._target_values(train)
        self.training_target = target
        self.global_bias = float(np.mean(target))
        item_sum = np.bincount(i, weights=target - self.global_bias, minlength=dataset.n_items)
        item_count = np.bincount(i, minlength=dataset.n_items)
        self.item_bias = item_sum / (item_count + 10.0)
        user_sum = np.bincount(u, weights=target - self.global_bias - self.item_bias[i], minlength=dataset.n_users)
        user_count = np.bincount(u, minlength=dataset.n_users)
        self.user_bias = user_sum / (user_count + 10.0)
        residual = target - self.global_bias - self.user_bias[u] - self.item_bias[i]

        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0.0, 0.05, (dataset.n_users, self.factors))
        self.item_factors = rng.normal(0.0, 0.05, (dataset.n_items, self.factors))
        user_rows = _group_rows(u, dataset.n_users)
        item_rows = _group_rows(i, dataset.n_items)
        ridge = self.reg * np.eye(self.factors)

        def loss() -> float:
            errors = residual - np.sum(self.user_factors[u] * self.item_factors[i], axis=1)
            return float(errors @ errors + self.reg * (
                np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)
            ))

        self.loss_history = [loss()]
        for _ in range(self.iterations):
            for user, rows in enumerate(user_rows):
                if len(rows) == 0:
                    continue
                matrix = self.item_factors[i[rows]]
                self.user_factors[user] = np.linalg.solve(matrix.T @ matrix + ridge, matrix.T @ residual[rows])
            for item, rows in enumerate(item_rows):
                if len(rows) == 0:
                    continue
                matrix = self.user_factors[u[rows]]
                self.item_factors[item] = np.linalg.solve(matrix.T @ matrix + ridge, matrix.T @ residual[rows])
            self.loss_history.append(loss())
        return self

    def score(self, user: int) -> np.ndarray:
        return (
            self.global_bias
            + self.user_bias[user]
            + self.item_bias
            + self.item_factors @ self.user_factors[user]
        )
