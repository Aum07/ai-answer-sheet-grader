"""Tune, ablate and train the model on real human-marked answers (Mohler et al. 2011).

1. Feature ablation: which feature groups actually help on real answers?
2. Regularisation: alpha is chosen inside each training fold (nested CV), never on test data.
3. The best feature set is retrained on ALL Mohler answers and saved to models/mohler/.

All evaluation is GroupKFold(5) by question, so every score is for questions the model
never saw. Uses the cached features from scripts/evaluate_mohler.py (no re-embedding).

    python scripts/tune_mohler.py         # -> reports/mohler_experiments.json, models/mohler/
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from grader import config  # noqa: E402
from grader.metrics import score_metrics  # noqa: E402
from scripts.evaluate_mohler import MOHLER, features_for, to10, with_paper_metrics  # noqa: E402

log = logging.getLogger("tune")
ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
GROUPS = {
    "meaning": ["Semantic Similarity", "question_student_similarity"],
    "length": ["Length Ratio"],
    "pos": [
        "POS_noun_ratio_diff",
        "POS_verb_ratio_diff",
        "POS_adj_ratio_diff",
        "POS_adv_ratio_diff",
        "POS_pos_diversity_diff",
        "POS_similarity",
    ],
    "readability": ["flesch_kincaid_ratio"],
    "coverage": config.COVERAGE_FEATURES,  # added after the adversarial tests
}
FEATURE_SETS = {
    "all_10_features": ["meaning", "length", "pos", "readability"],
    "drop_pos": ["meaning", "length", "readability"],
    "drop_readability": ["meaning", "length", "pos"],
    "drop_length": ["meaning", "pos", "readability"],
    "meaning_plus_length": ["meaning", "length"],
    "meaning_only": ["meaning"],
    "similarity_only": None,  # just answer-reference similarity
    "all_plus_coverage": ["meaning", "length", "pos", "readability", "coverage"],
    "drop_readability_plus_coverage": ["meaning", "length", "pos", "coverage"],
    "meaning_length_coverage": ["meaning", "length", "coverage"],
    "meaning_coverage": ["meaning", "coverage"],
}


def columns(set_name: str) -> list[str]:
    groups = FEATURE_SETS[set_name]
    if groups is None:
        return ["Semantic Similarity"]
    return [c for g in groups for c in GROUPS[g]]


def fit(X: pd.DataFrame, y: np.ndarray, alpha: float):
    from grader.ordinal import OrdinalLogisticAT

    scaler = StandardScaler().fit(X)  # fitted on a DataFrame: remembers feature names/order
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OrdinalLogisticAT(alpha=alpha, max_iter=3000).fit(scaler.transform(X), y)
    return model, scaler


def predict(model, scaler, X: pd.DataFrame) -> np.ndarray:
    return np.clip(np.round(model.predict(scaler.transform(X))), 0, 10).astype(int)


def qwk(y, p) -> float:
    return score_metrics(y, p)["qwk"]


def choose_alpha(X: pd.DataFrame, y: np.ndarray, groups: pd.Series, n_splits: int = 4) -> float:
    """Pick alpha by grouped CV on the data given (only ever called on training folds / full data)."""
    best_alpha, best_score = ALPHAS[0], -np.inf
    for alpha in ALPHAS:
        oof = np.zeros(len(y), dtype=int)
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            m, s = fit(X.iloc[tr], y[tr], alpha)
            oof[te] = predict(m, s, X.iloc[te])
        score = qwk(y, oof)
        if score > best_score:
            best_alpha, best_score = alpha, score
    return best_alpha


def main() -> None:
    if not MOHLER.exists():
        raise SystemExit("Run scripts/prepare_mohler.py first")
    df = features_for(MOHLER)
    df["Student Answer"] = df["Student Answer"].fillna("")
    y5 = df["human_avg"].to_numpy()
    y = to10(y5)
    groups = df["Question"].str.strip().str.lower()
    blank = df["Student Answer"].str.strip().eq("").to_numpy()
    outer = list(GroupKFold(n_splits=5).split(df, y, groups))
    log.info("Features computed on cleaned answers (repeats and filler sentences removed).")
    log.info("Mohler: %d answers, %d questions; nested GroupKFold 5x4, alphas %s\n", len(df), groups.nunique(), ALPHAS)

    human = {**score_metrics(to10(df["grader_1"]), to10(df["grader_2"])), **with_paper_metrics(df["grader_1"], to10(df["grader_2"]))}
    results: dict[str, dict] = {}
    oof_by_set: dict[str, np.ndarray] = {}
    for name in FEATURE_SETS:
        cols = columns(name)
        X = df[cols]
        oof = np.zeros(len(y), dtype=int)
        chosen = []
        for tr, te in outer:
            alpha = choose_alpha(X.iloc[tr], y[tr], groups.iloc[tr])
            m, s = fit(X.iloc[tr], y[tr], alpha)
            oof[te] = predict(m, s, X.iloc[te])
            chosen.append(alpha)
        oof[blank] = 0
        oof_by_set[name] = oof
        results[name] = {
            "features": cols,
            "alphas_chosen_per_fold": chosen,
            **score_metrics(y, oof),
            **with_paper_metrics(y5, oof),
        }
        r = results[name]
        log.info("%-22s %2d feats  QWK %.3f  r(0-5) %.3f  MAE %.2f  ±1 %4.1f%%  alphas %s",
                 name, len(cols), r["qwk"], r["pearson_0_5"], r["mae"], 100 * r["within_1"], chosen)

    log.info("%-22s           QWK %.3f  r(0-5) %.3f   (ceiling: grader 1 vs grader 2)", "human_vs_human", human["qwk"], human["pearson_0_5"])

    # Best = highest QWK; on a near-tie (< 0.005) prefer fewer features
    ranked = sorted(results, key=lambda n: (-round(results[n]["qwk"], 3), len(results[n]["features"])))
    best = ranked[0]
    for n in ranked[1:]:
        if results[best]["qwk"] - results[n]["qwk"] < 0.005 and len(results[n]["features"]) < len(results[best]["features"]):
            best = n
    cols = results[best]["features"]
    log.info("\nSelected feature set: %s (%d features)", best, len(cols))

    # Integrity checks on honest answers: how often do they fire, and what does capping cost?
    from grader.features import get_embedder
    from grader.integrity import CAP_FLAGS, FLAG_CAP, apply_caps, integrity_flags

    log.info("\nRunning integrity checks on all real answers...")
    originals = df["Original Answer"].fillna(df["Student Answer"]).astype(str).tolist()
    exercise = df["ID"].str[:3]  # an assignment = one "sheet" of questions
    sheet = df.groupby(exercise)["Reference Answer"].apply(lambda r: list(dict.fromkeys(r))).to_dict()
    flags = integrity_flags(
        df["Question"].tolist(), df["Reference Answer"].tolist(), originals, get_embedder(),
        all_references=[sheet[e] for e in exercise],
    )
    flagged = np.array([bool(CAP_FLAGS.intersection(f)) for f in flags])  # flags that change the score
    best_oof = oof_by_set[best]
    capped = apply_caps(best_oof, flags)
    capped[blank] = 0
    integrity = {
        "share_flagged": float(flagged.mean()),
        "by_flag": {k: float(np.mean([k in f for f in flags])) for k in ("copies_question", "keyword_list", "off_topic", "possibly_off_topic")},
        "human_mean_flagged": float(y[flagged].mean()) if flagged.any() else None,
        "human_mean_unflagged": float(y[~flagged].mean()),
        "qwk_without_caps": results[best]["qwk"],
        "qwk_with_caps": score_metrics(y, capped)["qwk"],
        "pearson_with_caps": with_paper_metrics(y5, capped)["pearson_0_5"],
    }
    log.info(
        "Capped %.1f%% of real answers (%s). Human mean score: capped %.2f vs others %.2f.",
        100 * integrity["share_flagged"],
        ", ".join(f"{k} {100 * v:.1f}%" for k, v in integrity["by_flag"].items()),
        integrity["human_mean_flagged"] or float("nan"), integrity["human_mean_unflagged"],
    )
    log.info("QWK with flag caps (max %d/10): %.3f (without: %.3f)", FLAG_CAP, integrity["qwk_with_caps"], integrity["qwk_without_caps"])

    # Final model: alpha chosen by grouped CV on ALL data, then fit on ALL data
    X = df[cols]
    alpha = choose_alpha(X, y, groups, n_splits=5)
    model, scaler = fit(X, y, alpha)
    out_dir = config.MODELS_DIR / "mohler"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "logistic_at.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")
    card = {
        "model": "grader.ordinal.OrdinalLogisticAT (mord.LogisticAT on label ranks)",
        "trained_on": "Mohler et al. 2011 (all 2,273 open-ended answers, 79 questions), human average 0-5 x 2",
        "source": "https://huggingface.co/datasets/nkazi/MohlerASAG (CC BY 4.0)",
        "features": cols,
        "alpha": alpha,
        "selection": "feature set and alpha chosen by question-grouped CV; metrics below are nested-CV estimates",
        "cv_metrics": {k: v for k, v in results[best].items() if k not in ("features",)},
        "human_vs_human": human,
        "integrity_checks_on_real_answers": integrity,
        "preprocessing": "answers cleaned (repeated / filler sentences removed) before features",
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    log.info("Saved final model (alpha=%s, trained on all data) -> %s", alpha, out_dir.relative_to(ROOT))

    report = ROOT / "reports/mohler_experiments.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"human_vs_human": human, "feature_sets": results, "selected": best, "final_alpha": alpha, "integrity": integrity}, indent=2))
    log.info("Wrote %s", report.relative_to(ROOT))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
