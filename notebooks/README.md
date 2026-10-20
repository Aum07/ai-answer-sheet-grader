# Notebooks

`01_original_research.ipynb` is the exploratory notebook from the original project (run on
Kaggle): feature engineering, comparison of ~12 classifiers/regressors (LogisticAT, XGBoost,
random forest, gradient boosting, TabNet, SVR, ...), hyperparameter search, bias analysis and
cross-validation.

It used a random split of answers, so answers to the same question appear in both train and
test. `scripts/train.py` is the reproducible version and splits by question instead.
