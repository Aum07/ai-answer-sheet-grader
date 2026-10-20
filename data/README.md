# Data

## `synthetic/synthetic_answers.csv`

4,500 rows, 1,496 unique questions, built for this project:

| Column | Meaning |
|---|---|
| `ID` | Question id |
| `Question` | Exam-style question (collected from the web or LLM-generated) |
| `Reference Answer` | LLM-written model answer |
| `Student Answer` | LLM-written answer imitating a student of a given quality |
| `Evaluation Score` | Target score 0–10 that the answer was written to deserve |
| `Semantic Similarity` | Legacy pre-computed similarity (recomputed by `scripts/build_features.py`) |

Score distribution is near-uniform (~400 answers per score). Because answers *and* labels come from an LLM, metrics on this data measure agreement with synthetic labels, not with human markers.

## Generated files (git-ignored)

- `processed/` – cached feature tables written by `scripts/build_features.py`
- `external/` – third-party datasets (e.g. human-marked short-answer datasets); download them yourself and respect their licences
