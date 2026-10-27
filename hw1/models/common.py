from __future__ import annotations

import numpy as np

def item_gram(dataset: Dataset) -> np.ndarray:
    gram = np.zeros((dataset.n_items, dataset.n_items), dtype=np.float64)
    for items in dataset.train.groupby("u")["i"]:
        unique = np.unique(items[1].to_numpy(dtype=int))
        gram[np.ix_(unique, unique)] += 1.0
    return gram
