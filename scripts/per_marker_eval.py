"""Every marker against every reference: teacher A, teacher B, both teachers' average.

Uses the saved predictions in data/processed/llm_predictions.csv (written by
scripts/evaluate_llm.py), so it needs no API calls and no retraining.

    python scripts/per_marker_eval.py      -> reports/per_marker_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from grader.metrics import score_metrics  # noqa: E402


def to10(x) -> np.ndarray:
    """0-5 human scores -> integer 0-10 labels (same as scripts/evaluate_mohler.py)."""
    return np.clip(np.round(np.asarray(x, dtype=float) * 2), 0, 10).astype(int)


def main() -> None:
    preds = pd.read_csv(ROOT / "data/processed/llm_predictions.csv")
    mohler = pd.read_csv(ROOT / "data/external/mohler_open_ended.csv")
    df = preds.merge(mohler[["ID", "grader_1", "grader_2", "human_avg"]], on="ID", how="left", validate="one_to_one")
    assert df[["grader_1", "grader_2"]].notna().all().all(), "unmatched IDs"

    markers = {
        "teacher_a": to10(df["grader_1"]),
        "teacher_b": to10(df["grader_2"]),
        "classic": df["classic"].to_numpy(),
        "llm": df["llm"].to_numpy(),
        "hybrid": np.round((df["llm"].to_numpy() + df["classic"].to_numpy()) / 2).astype(int),
    }
    references = {
        "teacher_a": markers["teacher_a"],
        "teacher_b": markers["teacher_b"],
        "teachers_average": to10(df["human_avg"]),
    }

    out = {"n_answers": len(df), "mean_score_0_10": {k: float(v.mean()) for k, v in {**markers, "teachers_average": references["teachers_average"]}.items()}, "results": {}}
    for ref_name, ref in references.items():
        out["results"][ref_name] = {}
        for name, pred in markers.items():
            if name == ref_name:
                continue
            out["results"][ref_name][name] = score_metrics(ref, pred)

    path = ROOT / "reports/per_marker_metrics.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
