"""Benchmark the LLM grader on real, human-marked answers (Mohler et al. 2011).

Compares, on the same answers and the same 0-10 scale:
  human_vs_human   grader 1 vs grader 2 (ceiling)
  classic_model    the shipped feature model, as question-grouped 5-fold CV predictions
  llm              the LLM grader (zero-shot: no Mohler examples in the prompt)
  hybrid           average of classic and LLM

    $env:OPENAI_API_KEY = "sk-..."              # PowerShell; never commit the key
    python scripts/evaluate_llm.py --questions 5     # cheap trial run (~150 answers)
    python scripts/evaluate_llm.py                   # all 79 questions
    python scripts/evaluate_llm.py --attacks         # + adversarial attacks on the LLM

Grades are cached in data/llm_cache/, so re-runs only pay for new answers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from grader import config  # noqa: E402
from grader import llm_grader  # noqa: E402
from grader.metrics import score_metrics  # noqa: E402
from scripts.evaluate_mohler import MOHLER, features_for, to10, with_paper_metrics  # noqa: E402
from scripts.tune_mohler import fit, predict  # noqa: E402

log = logging.getLogger("llm_eval")


def classic_oof(df_feats: pd.DataFrame, y10: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions of the shipped feature set (question-grouped 5-fold CV)."""
    card = json.loads((config.MODELS_DIR / "mohler" / "model_card.json").read_text())
    cols, alpha = card["features"], card["alpha"]
    groups = df_feats["Question"].str.strip().str.lower()
    oof = np.zeros(len(df_feats), dtype=int)
    for tr, te in GroupKFold(n_splits=5).split(df_feats, y10, groups):
        m, s = fit(df_feats.iloc[tr][cols], y10[tr], alpha)
        oof[te] = predict(m, s, df_feats.iloc[te][cols])
    return oof


WORKERS = 8  # parallel API requests (one per question)


def llm_scores(grader: llm_grader.LLMGrader, frame: pd.DataFrame, label: str) -> tuple[np.ndarray, list[str]]:
    from concurrent.futures import ThreadPoolExecutor

    scores = np.zeros(len(frame), dtype=int)
    feedback = [""] * len(frame)
    questions = list(frame.groupby(["Question", "Reference Answer"], sort=False))

    def work(item):
        (q, ref), part = item
        return part, grader.grade_question(q, ref, part["Student Answer"].fillna("").astype(str).tolist())

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = pool.map(work, questions)
    for k, (part, grades) in enumerate(results, 1):
        for pos, g in zip(part.index, grades):
            scores[frame.index.get_loc(pos)] = g.score
            feedback[frame.index.get_loc(pos)] = g.feedback
        if k % 10 == 0 or k == len(questions):
            u = grader.usage
            log.info("  %s: %d/%d questions  (calls %d, cached %d, ~$%.3f)", label, k, len(questions), u.calls, u.cached, u.cost_usd)
    return scores, feedback


def row(name: str, y10, y5, pred) -> dict:
    return {**score_metrics(y10, pred), **with_paper_metrics(y5, pred)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=int, help="only grade the first N questions (cheap trial)")
    ap.add_argument("--attacks", action="store_true", help="also run the adversarial attacks against the LLM")
    args = ap.parse_args()
    if not MOHLER.exists():
        raise SystemExit("Run scripts/prepare_mohler.py first")

    feats = features_for(MOHLER)  # same row order as the Mohler CSV
    raw = pd.read_csv(MOHLER)
    feats["Student Answer"] = raw["Student Answer"].fillna("").astype(str)  # LLM sees the original text
    y5_all = feats["human_avg"].to_numpy()
    y10_all = to10(y5_all)
    classic_all = classic_oof(feats, y10_all)
    classic_all[feats["Student Answer"].str.strip().eq("").to_numpy()] = 0

    subset = feats
    if args.questions:
        keep = list(dict.fromkeys(feats["Question"]))[: args.questions]
        subset = feats[feats["Question"].isin(keep)]
    idx = subset.index.to_numpy()
    y5, y10 = y5_all[idx], y10_all[idx]

    grader = llm_grader.LLMGrader()
    log.info("LLM grader: model %s, prompt %s; %d answers to %d questions",
             llm_grader.MODEL, llm_grader.PROMPT_VERSION, len(subset), subset["Question"].nunique())
    llm, feedback = llm_scores(grader, subset, "grading")
    classic = classic_all[idx]
    hybrid = np.round((llm + classic) / 2).astype(int)

    h1, h2 = to10(subset["grader_1"]), to10(subset["grader_2"])
    results = {
        "human_vs_human": {**score_metrics(h1, h2), **with_paper_metrics(subset["grader_1"], h2)},
        "classic_model": row("classic", y10, y5, classic),
        "llm": row("llm", y10, y5, llm),
        "hybrid_mean": row("hybrid", y10, y5, hybrid),
    }
    u = grader.usage
    usage = {
        "model": llm_grader.MODEL,
        "prompt_version": llm_grader.PROMPT_VERSION,
        "api_calls": u.calls,
        "cached_answers": u.cached,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "estimated_cost_usd": round(u.cost_usd, 4),
        "api_seconds": round(u.seconds, 1),
        "failed_answers": len(u.failures),
    }

    log.info("\n%-16s %6s %8s %6s %6s %6s", "vs human avg", "QWK", "r (0-5)", "MAE", "exact", "±1")
    for name, r in results.items():
        log.info("%-16s %6.3f %8.3f %6.2f %5.0f%% %5.0f%%", name, r["qwk"], r["pearson_0_5"], r["mae"], 100 * r["exact"], 100 * r["within_1"])
    log.info("\nMean score (0-10): humans %.2f | classic %.2f | LLM %.2f", y10.mean(), classic.mean(), llm.mean())
    log.info("LLM usage: %d calls, %d cached, %d+%d tokens, ~$%.3f, %.0fs of API time, %d failed",
             u.calls, u.cached, u.input_tokens, u.output_tokens, u.cost_usd, u.seconds, len(u.failures))

    attacks = None
    if args.attacks:
        from scripts.adversarial_eval import ATTACKS, attacked, sample_answers

        rng = np.random.default_rng(42)
        base = sample_answers(150, 42)
        orig, _ = llm_scores(grader, base, "original")
        attacks = {"original_mean": float(orig.mean()), "attacks": {}}
        for name, (_, expected, desc) in ATTACKS.items():
            s, _ = llm_scores(grader, attacked(base, name, rng), name)
            attacks["attacks"][name] = {
                "expected": expected,
                "mean_score": float(s.mean()),
                "mean_change": float((s - orig).mean()),
                "share_scored_8_plus": float((s >= 8).mean()),
            }
        log.info("\nLLM under attack (150 real answers humans scored <= 7/10); original mean %.2f", attacks["original_mean"])
        log.info("%-18s %-12s %8s %8s %8s", "attack", "expected", "mean", "change", "8+")
        for name, a in attacks["attacks"].items():
            log.info("%-18s %-12s %8.2f %+8.2f %7.0f%%", name, a["expected"], a["mean_score"], a["mean_change"], 100 * a["share_scored_8_plus"])
        usage["estimated_cost_usd"] = round(grader.usage.cost_usd, 4)

    out = ROOT / "reports/llm_metrics.json"
    out.write_text(json.dumps({"n_answers": len(subset), "n_questions": int(subset["Question"].nunique()),
                               "results": results, "usage": usage, "attacks": attacks}, indent=2))
    preds = subset[["ID", "Question", "Student Answer"]].assign(human_0_10=y10, classic=classic, llm=llm, llm_feedback=feedback)
    pred_path = ROOT / "data/processed/llm_predictions.csv"  # git-ignored; contains dataset text
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(pred_path, index=False)
    log.info("\nWrote %s and %s", out.relative_to(ROOT), pred_path.relative_to(ROOT))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
