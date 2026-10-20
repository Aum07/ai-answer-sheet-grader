import numpy as np
import pandas as pd
import pytest

from grader import config
from grader.extraction import NO_ANSWER
from grader.metrics import rescale, score_metrics

pytest.importorskip("nltk")
pytest.importorskip("textstat")

REF = (
    "Supervised learning trains a model on labelled examples so it can predict outputs for new inputs. "
    "Unsupervised learning finds structure such as clusters in unlabelled data. Reinforcement learning "
    "trains an agent with rewards from interacting with an environment."
)
QUESTION = "Explain supervised, unsupervised and reinforcement learning."


def frame(answers):
    return pd.DataFrame(
        {"ID": [str(i) for i in range(len(answers))], "Question": QUESTION, "Reference Answer": REF, "Student Answer": answers}
    )


def test_features_have_model_columns_and_sane_values(fake_embedder):
    from grader.features import compute_features

    out = compute_features(frame([REF, "I like football."]), embedder=fake_embedder)
    assert list(out[config.FEATURES].columns) == config.FEATURES
    same = out.iloc[0]
    assert same["Semantic Similarity"] == pytest.approx(1.0)
    assert same["Length Ratio"] == pytest.approx(1.0)
    assert same["POS_similarity"] == pytest.approx(1.0)
    assert out.iloc[1]["Semantic Similarity"] < 0.3


def test_shipped_model_ranks_good_answer_above_bad(fake_embedder):
    pytest.importorskip("mord")
    from grader.pipeline import grade_frame

    good = REF
    weak = "Supervised learning uses labels."
    junk = "I like football and pizza on weekends."
    from grader.scoring import load_model

    model, scaler = load_model(*config.model_paths("synthetic"))
    scored = grade_frame(frame([good, weak, junk, NO_ANSWER]), embedder=fake_embedder, model=model, scaler=scaler)
    s = scored["Predicted Score"].tolist()
    assert all(0 <= x <= 10 for x in s)
    assert s[0] >= s[1] and s[0] > s[2]  # full answer beats partial and off-topic ones
    assert s[3] == 0  # blank answers are not sent to the model


def test_metrics():
    m = score_metrics([0, 5, 10, 7], [0, 6, 10, 4])
    assert m["exact"] == 0.5 and m["within_1"] == 0.75
    assert m["mae"] == pytest.approx(1.0)
    assert np.allclose(rescale([0, 2.5, 5], from_max=5), [0, 5, 10])


def test_ordinal_model_handles_missing_scores():
    pytest.importorskip("mord")
    from grader.ordinal import OrdinalLogisticAT

    rng = np.random.default_rng(0)
    y = rng.choice([0, 2, 3, 5, 7, 10], size=300)  # gaps: no 1, 4, 6, 8, 9
    X = (y + rng.normal(0, 1, size=300)).reshape(-1, 1)
    pred = OrdinalLogisticAT().fit(X, y).predict(X)
    assert set(pred) <= {0, 2, 3, 5, 7, 10}
    assert np.corrcoef(pred, y)[0, 1] > 0.8
