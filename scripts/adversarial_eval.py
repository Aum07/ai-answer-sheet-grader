"""Adversarial robustness: does the grader reward substance or surface form?

Takes real Mohler answers that humans scored 7/10 or lower (so there is room to inflate),
applies each attack in grader/adversarial.py, and compares scores before and after for
every saved model, with the defences in grader/integrity.py switched off and on.

    python scripts/adversarial_eval.py               # 150 answers -> reports/adversarial.json
    python scripts/adversarial_eval.py --n 300
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from grader import config  # noqa: E402
from grader.adversarial import ATTACKS  # noqa: E402
from grader.features import compute_features  # noqa: E402
from grader.integrity import CAP_FLAGS, apply_caps, integrity_flags  # noqa: E402
from grader.scoring import load_model, predict_scores  # noqa: E402
from scripts.evaluate_mohler import MOHLER  # noqa: E402

log = logging.getLogger("adversarial")
MODELS = ["mohler", "synthetic"]


def sample_answers(n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(MOHLER)
    df["Student Answer"] = df["Student Answer"].fillna("").astype(str)
    df = df[(df["human_avg"] * 2 <= 7) & (df["Student Answer"].str.split().str.len() >= 5)]
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def attacked(base: pd.DataFrame, name: str, rng: np.random.Generator) -> pd.DataFrame:
    out = base[["ID", "Question", "Reference Answer", "Student Answer"]].copy()
    fn = ATTACKS[name][0]
    if name == "off_topic":
        # swap in a real answer to a *different* question from the same assignment (same sheet)
        full = pd.read_csv(MOHLER)
        full = full[full["Student Answer"].fillna("").str.split().str.len() >= 5]
        donors = []
        for qid, q in zip(out["ID"], out["Question"]):
            pool = full.loc[(full["ID"].str[:3] == qid[:3]) & (full["Question"] != q), "Student Answer"].to_numpy()
            if len(pool) == 0:
                pool = full.loc[full["Question"] != q, "Student Answer"].to_numpy()
            donors.append(pool[rng.integers(len(pool))])
        out["Student Answer"] = donors
    else:
        out["Student Answer"] = [
            fn(a, question=q, reference=r) for a, q, r in zip(out["Student Answer"], out["Question"], out["Reference Answer"])
        ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not MOHLER.exists():
        raise SystemExit("Run scripts/prepare_mohler.py first")

    rng = np.random.default_rng(args.seed)
    base = sample_answers(args.n, args.seed)
    models = {name: load_model(*config.model_paths(name)) for name in MODELS if config.model_paths(name)[0].exists()}
    log.info("%d real answers (human score <= 7/10); models: %s", len(base), ", ".join(models))

    frames = {"original": base[["ID", "Question", "Reference Answer", "Student Answer"]]}
    frames.update({name: attacked(base, name, rng) for name in ATTACKS})
    full = pd.read_csv(MOHLER)
    sheet = full.groupby(full["ID"].str[:3])["Reference Answer"].apply(lambda r: list(dict.fromkeys(r))).to_dict()
    refs = [sheet[i[:3]] for i in base["ID"]]  # each answer's sheet = its assignment's questions
    from grader.features import get_embedder

    emb = get_embedder()
    raw, clean, flags = {}, {}, {}
    for name, frame in frames.items():
        log.info("  embedding: %s", name)
        raw[name] = compute_features(frame, embedder=emb, clean=False)
        clean[name] = compute_features(frame, embedder=emb, clean=True)
        flags[name] = integrity_flags(
            frame["Question"].tolist(), frame["Reference Answer"].tolist(), frame["Student Answer"].tolist(), emb, all_references=refs
        )

    def summary(new, orig, fl, expected, desc):
        delta = new - orig
        return {
            "description": desc,
            "expected": expected,
            "mean_score": float(new.mean()),
            "mean_change": float(delta.mean()),
            "share_increased": float((delta > 0).mean()),
            "share_scored_8_plus": float((new >= 8).mean()),
            "share_flagged": float(np.mean([bool(f) for f in fl])) if fl is not None else None,
            "share_capped": float(np.mean([bool(CAP_FLAGS.intersection(f)) for f in fl])) if fl is not None else None,
        }

    results: dict[str, dict] = {}
    for mname, (model, scaler) in models.items():
        results[mname] = {}
        for mode in ("defences_off", "defences_on"):
            def score(name, mode=mode):
                if mode == "defences_off":
                    return predict_scores(raw[name], model, scaler)
                return apply_caps(predict_scores(clean[name], model, scaler), flags[name])

            orig = score("original")
            r = {"original_mean": float(orig.mean()), "attacks": {}}
            if mode == "defences_on":
                r["honest_answers_flagged"] = float(np.mean([bool(f) for f in flags["original"]]))
                r["honest_answers_capped"] = float(np.mean([bool(CAP_FLAGS.intersection(f)) for f in flags["original"]]))
            for aname, (_, expected, desc) in ATTACKS.items():
                r["attacks"][aname] = summary(score(aname), orig, flags[aname] if mode == "defences_on" else None, expected, desc)
            results[mname][mode] = r

    for mname, modes in results.items():
        off, on = modes["defences_off"], modes["defences_on"]
        log.info(
            "\nModel: %s   original mean %.2f (off) / %.2f (on); honest answers capped %.1f%%, flagged for review %.1f%%",
            mname, off["original_mean"], on["original_mean"], 100 * on["honest_answers_capped"], 100 * on["honest_answers_flagged"],
        )
        log.info("%-18s %-12s | %8s %9s | %8s %9s %8s %8s", "attack", "expected", "off:mean", "off:8+", "on:mean", "on:8+", "capped", "flagged")
        for aname in ATTACKS:
            a, b = off["attacks"][aname], on["attacks"][aname]
            log.info(
                "%-18s %-12s | %8.2f %8.0f%% | %8.2f %8.0f%% %7.0f%% %7.0f%%",
                aname, a["expected"], a["mean_score"], 100 * a["share_scored_8_plus"],
                b["mean_score"], 100 * b["share_scored_8_plus"], 100 * b["share_capped"], 100 * b["share_flagged"],
            )

    out = ROOT / "reports/adversarial.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_answers": len(base), "seed": args.seed, "results": results}, indent=2))
    log.info("\nWrote %s", out.relative_to(ROOT))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
