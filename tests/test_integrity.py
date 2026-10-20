import numpy as np
import pytest

from grader.integrity import (
    FLAG_CAP,
    apply_caps,
    clean_and_cover,
    copies_question,
    dedupe,
    integrity_flags,
    is_keyword_list,
    split_sentences,
)

Q = "What is a stack and how does it work?"
REF = "A stack stores items in LIFO order. Push adds an item on top. Pop removes the top item."
OTHER_REF = "A queue stores items in FIFO order. Enqueue adds at the rear. Dequeue removes from the front."


def test_split_and_dedupe():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert dedupe(["A stack is LIFO.", "a stack is lifo", "Push adds."]) == ["A stack is LIFO.", "Push adds."]


def test_cleaning_drops_filler_and_repeats(fake_embedder):
    answer = "A stack stores items in LIFO order. A stack stores items in LIFO order. Bananas grow quickly near warm rivers."
    cleaned, cov_mean, cov_frac = clean_and_cover([Q], [REF], [answer], fake_embedder)
    assert cleaned == ["A stack stores items in LIFO order."]
    assert 0 < cov_frac[0] < 1  # covered one of three reference points


def test_cleaning_never_empties_an_answer(fake_embedder):
    cleaned, _, _ = clean_and_cover([Q], [REF], ["Bananas grow quickly near warm rivers."], fake_embedder)
    assert cleaned == ["Bananas grow quickly near warm rivers."]


def test_coverage_rewards_covering_more_points(fake_embedder):
    partial = "A stack stores items in LIFO order."
    full = REF
    _, mean, frac = clean_and_cover([Q, Q], [REF, REF], [partial, full], fake_embedder)
    assert frac[1] > frac[0] and mean[1] > mean[0]


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("What is a stack and how does it work?", True),
        ("A stack: what is it and how does it work", True),
        ("A stack stores items in LIFO order; push adds and pop removes the top.", False),
    ],
)
def test_copies_question(answer, expected):
    assert copies_question(answer, Q) is expected


def test_keyword_list():
    assert is_keyword_list("stack items lifo order push adds item top removes")
    assert not is_keyword_list("A stack stores items in LIFO order and pop removes the top item.")


def test_off_topic_detected_against_other_references(fake_embedder):
    flags = integrity_flags(
        [Q, Q],
        [REF, REF],
        ["A stack stores items in LIFO order and pop removes the top item.", OTHER_REF],
        fake_embedder,
        all_references=[REF, OTHER_REF],
    )
    assert flags[0] == []
    assert "off_topic" in flags[1]


def test_caps_only_high_confidence_flags():
    scores = apply_caps(np.array([9, 9, 9]), [["off_topic"], ["possibly_off_topic"], []])
    assert scores.tolist() == [FLAG_CAP, 9, 9]  # review-only flags leave the score alone


def test_pipeline_flags_and_caps(fake_embedder):
    pytest.importorskip("mord")
    pytest.importorskip("nltk")
    import pandas as pd

    from grader import config
    from grader.pipeline import grade_frame
    from grader.scoring import load_model

    model, scaler = load_model(*config.model_paths("synthetic"))
    pairs = pd.DataFrame(
        {"ID": ["1", "2"], "Question": [Q, Q], "Reference Answer": [REF, REF], "Student Answer": [REF, Q]}
    )
    out = grade_frame(pairs, embedder=fake_embedder, model=model, scaler=scaler)
    assert out["Flags"].tolist()[0] == []
    assert "copies_question" in out["Flags"].tolist()[1]
    assert out["Predicted Score"].tolist()[1] <= FLAG_CAP
