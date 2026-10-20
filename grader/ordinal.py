"""Ordinal regression that tolerates gaps in the label set.

mord.LogisticAT requires labels to be exactly 0..K-1. Real marks often skip values
(e.g. nobody scored 1/10 in a training fold), so we fit on label ranks and map
predictions back to the original scores. The order of scores is preserved.
"""

from __future__ import annotations

import numpy as np


class OrdinalLogisticAT:
    def __init__(self, alpha: float = 1.0, max_iter: int = 3000):
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y):
        from mord import LogisticAT

        y = np.asarray(y).round().astype(int)
        self.classes_ = np.unique(y)  # sorted, e.g. [0, 2, 3, ..., 10]
        ranks = np.searchsorted(self.classes_, y)
        self.model_ = LogisticAT(alpha=self.alpha, max_iter=self.max_iter).fit(X, ranks)
        return self

    def predict(self, X) -> np.ndarray:
        ranks = np.clip(np.asarray(self.model_.predict(X)).astype(int), 0, len(self.classes_) - 1)
        return self.classes_[ranks]
