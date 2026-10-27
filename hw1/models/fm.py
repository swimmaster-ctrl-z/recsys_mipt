from __future__ import annotations

import numpy as np

def fm_raw_scores(features: np.ndarray, weights: np.ndarray, factors: np.ndarray, bias: float) -> np.ndarray:
    vectors = factors[features]
    summed = vectors.sum(axis=1)
    interactions = 0.5 * np.sum(summed * summed - np.sum(vectors * vectors, axis=1), axis=1)
    return bias + weights[features].sum(axis=1) + interactions


class FactorizationMachine:
    def __init__(
        self,
        factors: int = 16,
        epochs: int = 4,
        learning_rate: float = 0.15,
        regularization: float = 1e-4,
        negatives_per_positive: int = 1,
        batch_size: int = 1024,
        seed: int = 42,
    ):
        self.factors = factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.negatives_per_positive = negatives_per_positive
        self.batch_size = batch_size
        self.seed = seed

    def _make_features(self, dataset: Dataset) -> None:
        category_names = sorted({
            category
            for meta in dataset.item_metadata
            for category in meta["categories"]
        })
        store_names = sorted({meta["store"] for meta in dataset.item_metadata})
        start = 1 + dataset.n_users + dataset.n_items
        category_map = {name: start + offset for offset, name in enumerate(category_names)}
        start += len(category_names)
        store_map = {name: start + offset for offset, name in enumerate(store_names)}
        self.n_features = start + len(store_names)
        self.item_features = np.zeros((dataset.n_items, 4), dtype=np.int32)
        for item, meta in enumerate(dataset.item_metadata):
            for slot, category in enumerate(dict.fromkeys(meta["categories"])):
                if slot >= 3:
                    break
                self.item_features[item, slot] = category_map[category]
            self.item_features[item, 3] = store_map[meta["store"]]

    def feature_indices(self, user: int, item: int) -> np.ndarray:
        return self._feature_matrix(np.array([user]), np.array([item]))[0]

    def _feature_matrix(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return np.column_stack((
            users.astype(np.int32) + 1,
            items.astype(np.int32) + self.dataset.n_users + 1,
            self.item_features[items],
        ))

    def _training_examples(self, dataset: Dataset, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        train = dataset.train
        positive_users = train.u.to_numpy(dtype=int)
        positive_items = train.i.to_numpy(dtype=int)
        seen = train.groupby("u")["i"].agg(lambda x: set(x.tolist())).to_dict()
        candidates = np.arange(dataset.n_items)
        negative_users = []
        negative_items = []
        for user, count in train.groupby("u").size().items():
            available = np.setdiff1d(candidates, np.fromiter(seen[int(user)], dtype=int), assume_unique=False)
            if len(available) == 0:
                continue
            sampled = rng.choice(available, size=int(count) * self.negatives_per_positive, replace=True)
            negative_users.extend([int(user)] * len(sampled))
            negative_items.extend(sampled.tolist())
        users = np.concatenate((positive_users, np.asarray(negative_users, dtype=int)))
        items = np.concatenate((positive_items, np.asarray(negative_items, dtype=int)))
        labels = np.concatenate((np.ones(len(positive_users)), np.zeros(len(negative_users))))
        return self._feature_matrix(users, items), labels

    def fit(self, dataset: Dataset) -> "FactorizationMachine":
        self.dataset = dataset
        self._make_features(dataset)
        rng = np.random.default_rng(self.seed)
        features, labels = self._training_examples(dataset, rng)
        self.weights = np.zeros(self.n_features, dtype=np.float64)
        self.latent = rng.normal(0.0, 0.02, (self.n_features, self.factors))
        self.latent[0] = 0.0
        self.bias = 0.0
        weight_accumulator = np.zeros_like(self.weights)
        vector_accumulator = np.zeros_like(self.latent)
        self.loss_history = []

        for _ in range(self.epochs):
            order = rng.permutation(len(labels))
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                indices = features[batch]
                y = labels[batch]
                vectors = self.latent[indices]
                summed = vectors.sum(axis=1)
                logits = self.bias + self.weights[indices].sum(axis=1)
                logits += 0.5 * np.sum(summed * summed - np.sum(vectors * vectors, axis=1), axis=1)
                probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
                error = probabilities - y
                self.bias -= self.learning_rate * error.mean()
                unique, inverse = np.unique(indices.ravel(), return_inverse=True)
                weight_gradient = np.bincount(
                    inverse, weights=np.repeat(error, indices.shape[1]), minlength=len(unique)
                ) / len(batch)
                vector_gradient = np.zeros((len(unique), self.factors))
                example_gradient = (
                    error[:, None, None] * (summed[:, None, :] - vectors)
                ).reshape(-1, self.factors)
                np.add.at(vector_gradient, inverse, example_gradient)
                vector_gradient /= len(batch)
                weight_gradient += self.regularization * self.weights[unique]
                vector_gradient += self.regularization * self.latent[unique]
                weight_accumulator[unique] += weight_gradient ** 2
                vector_accumulator[unique] += vector_gradient ** 2
                self.weights[unique] -= self.learning_rate * weight_gradient / np.sqrt(weight_accumulator[unique] + 1e-8)
                self.latent[unique] -= self.learning_rate * vector_gradient / np.sqrt(vector_accumulator[unique] + 1e-8)
                self.weights[0] = 0.0
                self.latent[0] = 0.0
            logits = fm_raw_scores(features, self.weights, self.latent, self.bias)
            self.loss_history.append(float(np.mean(np.logaddexp(0.0, logits) - labels * logits)))
        return self

    def score(self, user: int) -> np.ndarray:
        items = np.arange(self.dataset.n_items)
        users = np.full(self.dataset.n_items, user, dtype=int)
        return fm_raw_scores(self._feature_matrix(users, items), self.weights, self.latent, self.bias)
