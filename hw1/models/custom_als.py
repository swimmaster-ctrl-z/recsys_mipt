from __future__ import annotations

import numpy as np
import pandas as pd

from .als import ExplicitALS

DAY_MS = 86_400_000.0

class CustomALS(ExplicitALS):
    def _target_values(self, train: pd.DataFrame) -> np.ndarray:
        ratings = train.rating.to_numpy(dtype=float)
        timestamps = train.timestamp.to_numpy(dtype=float)
        user_horizon = train.groupby("u").timestamp.transform("max").to_numpy(dtype=float)
        age_days = np.maximum(0.0, (user_horizon - timestamps) / DAY_MS)
        return np.maximum(0.0, (ratings - 3.0) / 2.0) * np.exp2(-age_days / 730.0)
