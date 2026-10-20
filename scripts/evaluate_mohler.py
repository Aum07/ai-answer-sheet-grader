"""Evaluate the grader on real, human-marked answers (Mohler et al. 2011).

Reports, on the same 0-10 scale:
  1. human_vs_human     grader 1 vs grader 2: the ceiling a model can realistically reach
  2. shipped_model      the model trained only on synthetic data, applied as-is (zero-shot transfer)
  3. similarity_only    a 1-feature baseline fitted on Mohler (question-grouped CV)
  4. trained_on_mohler  LogisticAT trained on Mohler itself (5-fold CV, grouped by question)
  5. synthetic+mohler   LogisticAT trained on synthetic data plus the Mohler training folds

Models 3-5 never see a test question during training (GroupKFold by question).
Pearson r and RMSE are also given on the original 0-5 scale, as in the Mohler paper.

    python scripts/prepare_mohler.py
    python scripts/evaluate_mohler.py           # -> reports/mohler_metrics.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from grader import config  # noqa: E402
from grader.metrics import score_metrics  # noqa: E402
from grader.scoring import load_model  # noqa: E402

from scripts.build_features import build, cache_path_for  # noqa: E402

log = logging.getLogger("mohler")
MOHLER = ROOT / "data/external/mohler_open_ended.csv"
SYNTHETIC = ROOT / "data/synthetic/synthetic_answers.csv"
ALPHA = 1.2747


def features_for(csv: Path) -> pd.DataFrame:
    """Cached features; rebuilt automatically if the cache predates newer feature columns."""
    cache = cache_path_for(csv)
    if cache.exists():
        df = pd.read_csv(cache)
        from grader.integrity import THRESHOLDS_PATH

        stale = THRESHOLDS_PATH.exists() and THRESHOLDS_PATH.stat().st_mtime > cache.stat().st_mtime
        if all(c in df.columns for c in config.ALL_FEATURES) and not stale:
            return df
        log.info("Feature cache %s is out of date (new columns or recalibrated cleaning); rebuilding", cache.name)
    build(csv, cache)
    return pd.read_csv(cache)


def to10(x) -> np.ndarray:
    """0-5 human scores -> integer 0-10 labels."""
    return np.clip(np.round(np.asarray(x, dtype=float) * 2), 0, 10).astype(int)


def with_paper_metrics(y5_true, pred10) -> dict:
    """Add Pearson / RMSE on the 0-5 scale for comparison with published Mohler results."""
    p5 = np.asarray(pred10, dtype=float) / 2
    t5 = np.asarray(y5_true, dtype=float)
    return {
        "pearson_0_5": float(np.corrcoef(t5, p5)[0, 1]) if p5.std() else 0.0,
        "rmse_0_5": float(np.sqrt(np.mean((t5 - p5) ** 2))),
    }


def logistic_at():
    from grader.ordinal import OrdinalLogisticAT

    return OrdinalLogisticAT(alpha=ALPHA, max_iter=3000)


def main() -> None:
    if not MOHLER.exists():
        raise SystemExit(f"{MOHLER} not found - run: python scripts/prepare_mohler.py")
    df = features_for(MOHLER)
    df["Student Answer"] = df["Student Answer"].fillna("")
    y5 = df["human_avg"].to_numpy()
    y10 = to10(y5)
    groups = df["Question"].str.strip().str.lower()
    X_all = df[config.FEATURES]
    blank = df["Student Answer"].str.strip().eq("").to_numpy()
    log.info("Mohler: %d answers, %d questions, %d blank", len(df), groups.nunique(), blank.sum())

    results: dict[str, dict] = {}

    # 1. Human-human agreement (the realistic ceiling)
    h1, h2 = to10(df["grader_1"]), to10(df["grader_2"])
    results["human_vs_human"] = {**score_metrics(h1, h2), **with_paper_metrics(df["grader_1"], h2)}

    # 2. Shipped model, zero-shot (trained on synthetic data only)
    model, scaler = load_model(*config.model_paths("synthetic"))  # zero-shot: synthetic-only model
    shipped = np.clip(np.round(model.predict(scaler.transform(X_all))), 0, 10).astype(int)
    shipped[blank] = 0
    results["shipped_model_zero_shot"] = {**score_metrics(y10, shipped), **with_paper_metrics(y5, shipped)}

    # 3-5. Question-grouped cross-validation on Mohler
    syn = features_for(SYNTHETIC).dropna(subset=config.FEATURES + ["Evaluation Score"])
    X_syn, y_syn = syn[config.FEATURES].to_numpy(), syn["Evaluation Score"].round().astype(int).clip(0, 10).to_numpy()
    preds = {k: np.zeros(len(df), dtype=int) for k in ("similarity_only", "trained_on_mohler", "synthetic_plus_mohler")}
    for train_idx, test_idx in GroupKFold(n_splits=5).split(X_all, y10, groups):
        Xtr, Xte, ytr = X_all.iloc[train_idx].to_numpy(), X_all.iloc[test_idx].to_numpy(), y10[train_idx]

        sim_tr = df["Semantic Similarity"].to_numpy()[train_idx]
        a, b = np.polyfit(sim_tr, ytr, 1)
        preds["similarity_only"][test_idx] = np.clip(np.round(a * df["Semantic Similarity"].to_numpy()[test_idx] + b), 0, 10)

        sc = StandardScaler().fit(Xtr)
        m = logistic_at().fit(sc.transform(Xtr), ytr)
        preds["trained_on_mohler"][test_idx] = np.clip(m.predict(sc.transform(Xte)), 0, 10)

        Xboth, yboth = np.vstack([X_syn, Xtr]), np.concatenate([y_syn, ytr])
        sc2 = StandardScaler().fit(Xboth)
        m2 = logistic_at().fit(sc2.transform(Xboth), yboth)
        preds["synthetic_plus_mohler"][test_idx] = np.clip(m2.predict(sc2.transform(Xte)), 0, 10)

    for name, p in preds.items():
        p = p.copy()
        p[blank] = 0
        results[name] = {**score_metrics(y10, p), **with_paper_metrics(y5, p)}

    # Score distribution: does the synthetic model compress or shift real scores?
    dist = {
        "human_avg_0_10_mean": float(y10.mean()),
        "shipped_model_mean": float(shipped.mean()),
        "human_share_8_plus": float((y10 >= 8).mean()),
        "shipped_share_8_plus": float((shipped >= 8).mean()),
    }

    log.info("\n%-26s %6s %6s %6s %6s %8s", "vs human average (0-10)", "QWK", "MAE", "exact", "±1", "r (0-5)")
    for name, r in results.items():
        log.info(
            "%-26s %6.3f %6.2f %5.1f%% %5.1f%% %8.3f",
            name, r["qwk"], r["mae"], 100 * r["exact"], 100 * r["within_1"], r["pearson_0_5"],
        )
    log.info("\nMean score: humans %.2f vs shipped model %.2f (0-10)", dist["human_avg_0_10_mean"], dist["shipped_model_mean"])
    log.info("Share scored 8+: humans %.0f%% vs shipped model %.0f%%", 100 * dist["human_share_8_plus"], 100 * dist["shipped_share_8_plus"])

    out = ROOT / "reports/mohler_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "data": "Mohler et al. 2011 via huggingface.co/datasets/nkazi/MohlerASAG (raw, open_ended)",
                "n_answers": int(len(df)),
                "n_questions": int(groups.nunique()),
                "cv": "GroupKFold(5) by question for trained models",
                "results": results,
                "distribution": dist,
            },
            indent=2,
        )
    )
    log.info("\nWrote %s", out.relative_to(ROOT))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
