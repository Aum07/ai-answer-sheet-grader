"""Central configuration. Every value can be overridden with an environment variable."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Which trained model the app uses: a folder under models/ ("mohler" = trained on real,
# human-marked answers; "synthetic" = trained on the LLM-generated dataset).
MODEL_NAME = os.getenv("MODEL_NAME", "mohler")
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / MODEL_NAME / "logistic_at.pkl"))
SCALER_PATH = Path(os.getenv("SCALER_PATH", MODELS_DIR / MODEL_NAME / "scaler.pkl"))


def model_paths(name: str) -> tuple[Path, Path]:
    """(model, scaler) paths for a named model folder, e.g. model_paths("synthetic")."""
    return MODELS_DIR / name / "logistic_at.pkl", MODELS_DIR / name / "scaler.pkl"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE") or None  # None = auto (GPU if available); "cpu" to force

MAX_SCORE = 10  # scores are integers 0..MAX_SCORE per question

# API settings
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", 10))
MAX_STUDENT_FILES = int(os.getenv("MAX_STUDENT_FILES", 30))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 10))
API_KEY = os.getenv("API_KEY", "")  # empty = auth disabled (local/demo use)
RESULT_TTL_SECONDS = int(os.getenv("RESULT_TTL_SECONDS", 3600))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

# Feature order is part of the model contract: the scaler and model were fitted on
# columns in exactly this order. Do not reorder without retraining.
FEATURES = [
    "Semantic Similarity",
    "Length Ratio",
    "question_student_similarity",
    "POS_noun_ratio_diff",
    "POS_similarity",
    "flesch_kincaid_ratio",
    "POS_pos_diversity_diff",
    "POS_verb_ratio_diff",
    "POS_adv_ratio_diff",
    "POS_adj_ratio_diff",
]
# Added after the adversarial tests: share of reference points covered, sentence by sentence.
COVERAGE_FEATURES = ["coverage_mean", "coverage_frac"]
ALL_FEATURES = FEATURES + COVERAGE_FEATURES
