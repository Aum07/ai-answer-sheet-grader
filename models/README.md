# Models

Each folder holds one trained ordinal model (0–10, mord LogisticAT), its `StandardScaler`
and a `model_card.json` with the training data, features, and held-out metrics.
`integrity.json` holds the calibrated thresholds for the answer-cleaning and integrity checks
(`scripts/calibrate_integrity.py`).

| Folder | Trained on | Use |
|---|---|---|
| `mohler/` | 2,273 real student answers marked by two humans (Mohler et al. 2011, CS / data structures) | **Default.** Best on real answers; tuned on computer-science questions |
| `synthetic/` | 4,500 LLM-generated answers across many subjects | Kept for comparison; scores real answers poorly (see README) |

Pick one with the `MODEL_NAME` environment variable (default `mohler`).
Rebuild `mohler/` with `python scripts/tune_mohler.py` (feature selection + nested CV, final fit on all answers) and `synthetic/` with `python scripts/train.py`.
Each model may use a subset of the 10 features; the scaler stores which ones (`feature_names_in_`).

Joblib pickles made with scikit-learn 1.6.1 / numpy 1.26 (see `requirements.txt`).
Load them only from this repository; pickles can execute code.
