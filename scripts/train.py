"""Train the ordinal scoring model reproducibly and report honest metrics.

Splits are made by *question* (GroupShuffleSplit), so no question seen in training
appears in validation or test. This measures what matters in production: grading
questions the model has never seen. Answers are scored 0..10.

    python scripts/train.py                         # synthetic data -> models/synthetic/
    python scripts/train.py --dataset mohler        # real data, 70/15/15 report only (ship via tune_mohler.py)
    python scripts/train.py --no-save               # just report metrics

Outputs:
    models/<name>/logistic_at.pkl, scaler.pkl, model_card.json   (unless --no-save)
    reports/metrics_<name>.json                                   (all models, all splits)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from grader import config  # noqa: E402
from grader.metrics import score_metrics  # noqa: E402
from grader.ordinal import OrdinalLogisticAT  # noqa: E402
from scripts.build_features import build, cache_path_for  # noqa: E402

log = logging.getLogger("train")
SEED = 42

DATASETS = {
    "synthetic": {"data": ROOT / "data/synthetic/synthetic_answers.csv", "score_col": "Evaluation Score", "multiplier": 1.0},
    # Real student answers; run scripts/prepare_mohler.py first. Human average 0-5 -> 0-10.
    "mohler": {"data": ROOT / "data/external/mohler_open_ended.csv", "score_col": "human_avg", "multiplier": 2.0},
}


def split_by_question(df: pd.DataFrame, seed: int = SEED):
    """70/15/15 train/val/test split with whole questions kept together."""
    groups = df["Question"].str.strip().str.lower()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, rest_idx = next(outer.split(df, groups=groups))
    rest = df.iloc[rest_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(inner.split(rest, groups=groups.iloc[rest_idx]))
    return df.iloc[train_idx], rest.iloc[val_rel], rest.iloc[test_rel]


def candidate_models(alpha: float) -> dict:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    return {
        "logistic_at": OrdinalLogisticAT(alpha=alpha, max_iter=3000),  # ordinal: respects 3 < 4 < 5
        "ridge_rounded": Ridge(alpha=1.0),
        "gbr_rounded": GradientBoostingRegressor(random_state=SEED),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="synthetic", help="preset for --data/--score-col")
    ap.add_argument("--data", type=Path, help="override the dataset CSV")
    ap.add_argument("--score-col", help="override the score column")
    ap.add_argument("--score-multiplier", type=float, help="scale scores to 0-10 (Mohler is 0-5 -> 2)")
    ap.add_argument("--alpha", type=float, default=1.2747, help="LogisticAT L2 strength (tuned on original split)")
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    preset = DATASETS[args.dataset]
    args.data = args.data or preset["data"]
    args.score_col = args.score_col or preset["score_col"]
    mult = args.score_multiplier or preset["multiplier"]
    args.report = args.report or ROOT / f"reports/metrics_{args.dataset}.json"

    cache = cache_path_for(args.data)
    if args.rebuild_features or not cache.exists():
        build(args.data, cache)
    df = pd.read_csv(cache)
    df = df.dropna(subset=config.FEATURES + [args.score_col])
    y_all = (df[args.score_col] * mult).round().astype(int).clip(0, config.MAX_SCORE)
    df = df.assign(_y=y_all)

    train, val, test = split_by_question(df)
    log.info(
        "Rows train/val/test: %d/%d/%d  (questions %d/%d/%d)",
        len(train), len(val), len(test),
        train["Question"].nunique(), val["Question"].nunique(), test["Question"].nunique(),
    )

    scaler = StandardScaler().fit(train[config.FEATURES])
    X = {name: scaler.transform(part[config.FEATURES]) for name, part in [("train", train), ("val", val), ("test", test)]}
    y = {"train": train["_y"].to_numpy(), "val": val["_y"].to_numpy(), "test": test["_y"].to_numpy()}

    results: dict[str, dict] = {}

    # Baselines: anything worth shipping must clearly beat these.
    mean_score = round(float(y["train"].mean()))
    results["baseline_mean"] = {s: score_metrics(y[s], np.full_like(y[s], mean_score)) for s in ("val", "test")}
    sim = {s: part["Semantic Similarity"].to_numpy() for s, part in [("train", train), ("val", val), ("test", test)]}
    a, b = np.polyfit(sim["train"], y["train"], 1)  # linear map from cosine similarity to score
    results["baseline_similarity_only"] = {
        s: score_metrics(y[s], np.clip(np.round(a * sim[s] + b), 0, config.MAX_SCORE)) for s in ("val", "test")
    }

    fitted = {}
    for name, model in candidate_models(args.alpha).items():
        model.fit(X["train"], y["train"])
        fitted[name] = model
        results[name] = {
            s: score_metrics(y[s], np.clip(np.round(model.predict(X[s])), 0, config.MAX_SCORE))
            for s in ("train", "val", "test")
        }

    # Show the table
    log.info("\n%-26s %6s %6s %6s %6s", "model (test split)", "QWK", "MAE", "exact", "±1")
    for name, r in results.items():
        t = r["test"]
        log.info("%-26s %6.3f %6.2f %6.1f%% %5.1f%%", name, t["qwk"], t["mae"], 100 * t["exact"], 100 * t["within_1"])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "data": str(args.data.relative_to(ROOT) if args.data.is_relative_to(ROOT) else args.data),
        "split": "GroupShuffleSplit by question, 70/15/15, seed 42",
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "questions": {"train": int(train["Question"].nunique()), "val": int(val["Question"].nunique()), "test": int(test["Question"].nunique())},
        "features": config.FEATURES,
        "logistic_at_alpha": args.alpha,
        "score_column": args.score_col,
        "score_multiplier": mult,
    }
    args.report.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    log.info("\nWrote %s", args.report)

    if args.dataset == "mohler" and not args.no_save:
        log.info("\nNot saving: the shipped Mohler model is built by scripts/tune_mohler.py "
                 "(nested CV, feature selection, trained on all answers).")
        args.no_save = True
    if not args.no_save:
        model_path, scaler_path = config.model_paths(args.dataset)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted["logistic_at"], model_path)
        joblib.dump(scaler, scaler_path)
        card = {**meta, "model": "mord.LogisticAT", "test_metrics": results["logistic_at"]["test"]}
        (model_path.parent / "model_card.json").write_text(json.dumps(card, indent=2))
        log.info("Saved model -> %s", model_path.parent)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
