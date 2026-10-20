"""Agreement metrics for ordinal scores (as used in automated essay/short-answer scoring)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score


def score_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    return {
        "n": int(len(y_true)),
        "qwk": float(cohen_kappa_score(y_true.round().astype(int), y_pred.round().astype(int), weights="quadratic")),
        "mae": float(err.mean()),
        "exact": float((err < 0.5).mean()),
        "within_1": float((err <= 1).mean()),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.std() and y_pred.std() else 0.0,
    }


def rescale(scores, from_max: float, to_max: float = 10) -> np.ndarray:
    """Map scores from a 0..from_max scale onto 0..to_max (e.g. Mohler 0-5 -> 0-10)."""
    return np.asarray(scores, dtype=float) * (to_max / from_max)
