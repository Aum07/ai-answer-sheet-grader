"""Calibrate the integrity-check thresholds on real answers, at fixed false-positive rates.

Honest = real Mohler answers. Positives = the same answers paired with a *different* question
from the same assignment (what "answered the wrong question" looks like on a real sheet), and
generic filler sentences (a set separate from the one the adversarial test uses).

Thresholds are chosen on half of the questions and reported on the other half, then saved to
models/integrity.json, which grader/integrity.py loads at runtime.

    python scripts/calibrate_integrity.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from grader import integrity  # noqa: E402
from grader.features import get_embedder  # noqa: E402
from scripts.evaluate_mohler import MOHLER  # noqa: E402

log = logging.getLogger("calibrate")
CAP_FP, REVIEW_FP, KEEP_RATE = 0.01, 0.05, 0.98

# Calibration filler: deliberately different wording from grader.adversarial.FILLER
CALIBRATION_FILLER = [
    "This question is an interesting one and it is worth thinking about carefully.",
    "There are several ways of looking at this problem in general.",
    "Many people have different opinions about this subject.",
    "It depends on the situation and the context in which it is used.",
    "This is something that every student should know about.",
    "In my opinion this is one of the most important ideas we studied.",
    "Overall, I think the answer is fairly straightforward once you understand it.",
    "As discussed in class, this topic comes up quite often.",
    "To sum up, there are advantages and disadvantages to consider.",
    "It is important to remember the basic principles here.",
    "Different textbooks explain this in slightly different ways.",
    "This concept has many practical applications in the real world.",
    "I hope this explanation answers the question well enough.",
    "Experts agree that this area is very significant.",
    "The details can be complicated but the main idea is simple.",
    "Further study would make this topic even clearer.",
]


def unit(m):
    m = np.asarray(m, dtype=float)
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)


def main() -> None:
    if not MOHLER.exists():
        raise SystemExit("Run scripts/prepare_mohler.py first")
    df = pd.read_csv(MOHLER)
    df["Student Answer"] = df["Student Answer"].fillna("").astype(str)
    df = df[df["Student Answer"].str.strip().ne("")].reset_index(drop=True)
    df["exercise"] = df["ID"].str[:3]
    groups = df["Question"].str.strip().str.lower()
    calib_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=0).split(df, groups=groups))
    rng = np.random.default_rng(0)
    emb = get_embedder()

    # ---- whole-answer embeddings for the off-topic check
    log.info("Embedding %d answers and %d references...", len(df), df["Reference Answer"].nunique())
    A = unit(emb.encode(df["Student Answer"].tolist(), batch_size=64, show_progress_bar=False))
    refs = df["Reference Answer"].unique().tolist()
    R = dict(zip(refs, unit(emb.encode(refs, show_progress_bar=False))))
    sheet = df.groupby("exercise")["Reference Answer"].apply(lambda s: list(dict.fromkeys(s))).to_dict()

    def own_and_margin(i: int, own_ref: str):
        a = A[i]
        own = float(a @ R[own_ref])
        others = [float(a @ R[r]) for r in sheet[df.at[i, "exercise"]] if r != own_ref]
        return own, (max(others) - own) if others else -1.0

    honest = np.array([own_and_margin(i, df.at[i, "Reference Answer"]) for i in range(len(df))])
    swapped = []  # same answer graded against another question's reference on the same sheet
    for i in range(len(df)):
        pool = [r for r in sheet[df.at[i, "exercise"]] if r != df.at[i, "Reference Answer"]]
        if not pool:
            swapped.append((np.nan, np.nan))
            continue
        wrong = pool[rng.integers(len(pool))]
        a = A[i]
        own = float(a @ R[wrong])
        others = [float(a @ R[r]) for r in sheet[df.at[i, "exercise"]] if r != wrong]
        swapped.append((own, max(others) - own))
    swapped = np.array(swapped)

    def fires(vals, abs_t, margin_t):
        own, margin = vals[:, 0], vals[:, 1]
        return (own < abs_t) | (margin > margin_t)

    def pick(target_fp, idx):
        best = None
        h, s = honest[idx], swapped[idx]
        s = s[~np.isnan(s[:, 0])]
        for abs_t in np.arange(0.0, 0.51, 0.025):
            for margin_t in np.arange(0.0, 0.61, 0.02):
                fp = fires(h, abs_t, margin_t).mean()
                if fp <= target_fp:
                    tpr = fires(s, abs_t, margin_t).mean()
                    if best is None or tpr > best[2]:
                        best = (float(abs_t), float(margin_t), float(tpr), float(fp))
        return best

    def report(abs_t, margin_t, idx):
        h, s = honest[idx], swapped[idx]
        s = s[~np.isnan(s[:, 0])]
        return {"false_positive_rate": float(fires(h, abs_t, margin_t).mean()), "detection_rate": float(fires(s, abs_t, margin_t).mean())}

    cap = pick(CAP_FP, calib_idx)
    review = pick(REVIEW_FP, calib_idx)

    # ---- sentence relevance threshold for cleaning (filler removal)
    log.info("Embedding sentences for the filler threshold...")
    rows = []
    for i in range(len(df)):
        anchors = integrity.split_sentences(df.at[i, "Reference Answer"]) + [df.at[i, "Question"]]
        for s in integrity.split_sentences(df.at[i, "Student Answer"]):
            rows.append((i, s, anchors, 0))
        rows.append((i, CALIBRATION_FILLER[rng.integers(len(CALIBRATION_FILLER))], anchors, 1))
    cache = integrity.SentenceCache(emb)
    cache.add([s for _, s, _, _ in rows] + [a for _, _, an, _ in rows for a in an])
    rel = np.array([float((cache.matrix([s]) @ cache.matrix(an).T).max()) for _, s, an, _ in rows])
    is_filler = np.array([f for *_, f in rows], dtype=bool)
    row_idx = np.array([i for i, *_ in rows])
    in_calib = np.isin(row_idx, calib_idx)
    relevance_min = float(np.quantile(rel[in_calib & ~is_filler], 1 - KEEP_RATE))
    in_test = ~in_calib

    # ---- keyword-list check: false positives on honest answers by minimum length
    kw_fp = {n: float(np.mean([integrity.is_keyword_list(a, min_words=n) for a in df["Student Answer"]])) for n in (4, 5, 6)}
    kw_min = min((n for n, fp in kw_fp.items() if fp <= 0.01), default=6)
    cq_fp = float(np.mean([integrity.copies_question(a, q) for a, q in zip(df["Student Answer"], df["Question"])]))

    thresholds = {
        "relevance_min": round(relevance_min, 4),
        "off_topic_cap": {"abs": cap[0], "margin": cap[1]},
        "off_topic_review": {"abs": review[0], "margin": review[1]},
        "keyword_min_words": kw_min,
    }
    evaluation = {
        "note": "thresholds chosen on half the questions (calibration), reported on the other half (test)",
        "off_topic_cap_test": report(cap[0], cap[1], test_idx),
        "off_topic_review_test": report(review[0], review[1], test_idx),
        "cleaning_test": {
            "honest_sentences_kept": float((rel[in_test & ~is_filler] >= relevance_min).mean()),
            "filler_sentences_removed": float((rel[in_test & is_filler] < relevance_min).mean()),
        },
        "keyword_list_false_positive_by_min_words": kw_fp,
        "copies_question_false_positive": cq_fp,
    }
    out = integrity.THRESHOLDS_PATH
    out.write_text(json.dumps({"thresholds": thresholds, "evaluation": evaluation}, indent=2))

    log.info("\nThresholds (chosen on calibration half):")
    log.info("  filler: drop sentences with relevance < %.3f", relevance_min)
    log.info("  off-topic, cap:    own < %.3f or margin > %.2f", *cap[:2])
    log.info("  off-topic, review: own < %.3f or margin > %.2f", *review[:2])
    log.info("  keyword list: >= %d words with <10%% function words", kw_min)
    log.info("\nOn held-out questions:")
    for k in ("off_topic_cap_test", "off_topic_review_test"):
        r = evaluation[k]
        log.info("  %-22s honest wrongly flagged %5.1f%%   wrong-question answers caught %5.1f%%", k, 100 * r["false_positive_rate"], 100 * r["detection_rate"])
    c = evaluation["cleaning_test"]
    log.info("  cleaning               honest sentences kept %5.1f%%   filler removed %5.1f%%", 100 * c["honest_sentences_kept"], 100 * c["filler_sentences_removed"])
    log.info("  keyword-list false positives by min words: %s", {k: f"{100 * v:.1f}%" for k, v in kw_fp.items()})
    log.info("  copies-question false positives: %.1f%%", 100 * cq_fp)
    log.info("\nWrote %s", out.relative_to(ROOT))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
