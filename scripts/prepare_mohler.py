"""Download the Mohler et al. (2011) short-answer dataset and convert it to this project's CSV format.

Real student answers from a University of North Texas data-structures course, each scored
0-5 by two human graders. Source: https://huggingface.co/datasets/nkazi/MohlerASAG (CC BY 4.0).

    pip install datasets
    python scripts/prepare_mohler.py          # -> data/external/mohler_open_ended.csv

Citation: Mohler, Bunescu & Mihalcea (2011). Learning to Grade Short Answer Questions using
Semantic Similarity Measures and Dependency Graph Alignments. ACL. https://aclanthology.org/P11-1076
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "external" / "mohler_open_ended.csv"
TEN_POINT_EXERCISES = ("E11", "E12")  # graded 0-10 in the original data; everything else 0-5


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the Hugging Face datasets library first:  pip install datasets") from exc

    ds = load_dataset("nkazi/MohlerASAG", name="raw", split="open_ended")
    df = ds.to_pandas()

    # Put both graders on a 0-5 scale (E11/E12 were marked out of 10)
    ten = df["id"].str.startswith(TEN_POINT_EXERCISES)
    for col in ("score_grader_1", "score_grader_2"):
        df.loc[ten, col] = df.loc[ten, col] / 2

    out = pd.DataFrame(
        {
            "ID": df["id"],
            "Question": df["question"].str.strip(),
            "Reference Answer": df["instructor_answer"].str.strip(),
            "Student Answer": df["student_answer"].fillna("").str.strip(),
            "grader_1": df["score_grader_1"],  # 0-5
            "grader_2": df["score_grader_2"],  # 0-5
            "human_avg": df["score_avg"],  # 0-5
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(
        f"Wrote {OUT.relative_to(ROOT)}: {len(out)} answers, {out['Question'].nunique()} questions, "
        f"grader_1 range {out.grader_1.min()}-{out.grader_1.max()}, grader_2 range {out.grader_2.min()}-{out.grader_2.max()}"
    )


if __name__ == "__main__":
    main()
