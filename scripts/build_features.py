"""Compute model features for a dataset CSV once and cache them (embedding is the slow part).

Input CSV columns: Question, Reference Answer, Student Answer, plus a score column.

    python scripts/build_features.py data/synthetic/synthetic_answers.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grader.features import compute_features  # noqa: E402

REQUIRED = ["Question", "Reference Answer", "Student Answer"]


def cache_path_for(csv: Path) -> Path:
    return csv.parent.parent / "processed" / f"{csv.stem}_features.csv"


def build(csv: Path, out: Path | None = None) -> Path:
    df = pd.read_csv(csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv} is missing columns: {missing}")
    for col in REQUIRED:
        df[col] = df[col].fillna("").astype(str)
    out = out or cache_path_for(csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Computing features for %d rows (this takes a few minutes on CPU)...", len(df))
    compute_features(df).to_csv(out, index=False)
    logging.info("Wrote %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    build(args.csv, args.out)
