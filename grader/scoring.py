"""Load the trained ordinal model and turn feature rows into 0-10 scores."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config


@lru_cache(maxsize=4)
def load_model(model_path: str | Path = config.MODEL_PATH, scaler_path: str | Path = config.SCALER_PATH):
    """Load (model, scaler) once per process; joblib files are trusted local artefacts only."""
    return joblib.load(model_path), joblib.load(scaler_path)


def predict_scores(features: pd.DataFrame, model=None, scaler=None) -> np.ndarray:
    if model is None or scaler is None:
        model, scaler = load_model()
    if features.empty:
        return np.array([], dtype=int)
    # Each model may use a subset of the features; the scaler remembers which (and their order).
    cols = list(getattr(scaler, "feature_names_in_", config.FEATURES))
    X = scaler.transform(features[cols])
    preds = np.asarray(model.predict(X)).round().astype(int)
    return np.clip(preds, 0, config.MAX_SCORE)


def models_available() -> bool:
    return Path(config.MODEL_PATH).exists() and Path(config.SCALER_PATH).exists()
