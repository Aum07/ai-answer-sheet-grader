"""Hand-crafted features comparing a student answer with the reference answer.

These must stay identical to the features the shipped model was trained on
(see `config.FEATURES`). The embedding model is injected so tests and
experiments can swap it for a fake or a different encoder.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

POS_GROUPS = {
    "noun_ratio": ("NN", "NNS", "NNP", "NNPS"),
    "verb_ratio": ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"),
    "adj_ratio": ("JJ", "JJR", "JJS"),
    "adv_ratio": ("RB", "RBR", "RBS"),
}
POS_KEYS = [*POS_GROUPS, "pos_diversity"]


class Embedder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs) -> np.ndarray: ...


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s", config.EMBEDDING_MODEL)
    return SentenceTransformer(config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE)


@lru_cache(maxsize=1)
def _ensure_nltk() -> None:
    import nltk

    # Older NLTK uses punkt / averaged_perceptron_tagger, NLTK >= 3.8.2 the *_tab/_eng variants.
    for resource, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def pos_profile(text: str) -> dict[str, float]:
    """Share of nouns/verbs/adjectives/adverbs and tag diversity in a text."""
    from nltk import pos_tag, word_tokenize

    _ensure_nltk()
    tokens = word_tokenize(text.lower())
    if not tokens:
        return dict.fromkeys(POS_KEYS, 0.0)
    counts: dict[str, int] = {}
    for _, tag in pos_tag(tokens):
        counts[tag] = counts.get(tag, 0) + 1
    n = len(tokens)
    profile = {name: sum(counts.get(t, 0) for t in tags) / n for name, tags in POS_GROUPS.items()}
    profile["pos_diversity"] = len(counts) / n
    return profile


def flesch_kincaid_ratio(reference: str, student: str) -> float:
    import textstat

    ref = textstat.flesch_kincaid_grade(reference)
    return textstat.flesch_kincaid_grade(student) / ref if ref else 0.0


def _rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return np.sum(a * b, axis=1)


def compute_features(
    df: pd.DataFrame, embedder: Embedder | None = None, batch_size: int = 32, clean: bool = True
) -> pd.DataFrame:
    """Add feature columns to a frame with Question / Reference Answer / Student Answer.

    With `clean=True`, repeated and off-topic (filler) sentences are removed first and every
    feature is computed on what remains; the untouched text is kept in "Original Answer".
    """
    from .integrity import clean_and_cover

    df = df.copy()
    if df.empty:
        for col in config.ALL_FEATURES + ["Original Answer"]:
            df[col] = pd.Series(dtype=float)
        return df

    embedder = embedder or get_embedder()
    kw = {"batch_size": batch_size, "show_progress_bar": False}
    df["Original Answer"] = df["Student Answer"]
    cleaned, cov_mean, cov_frac = clean_and_cover(
        df["Question"].tolist(), df["Reference Answer"].tolist(), df["Student Answer"].tolist(), embedder, clean=clean
    )
    if clean:
        df["Student Answer"] = cleaned
    df["coverage_mean"] = cov_mean
    df["coverage_frac"] = cov_frac
    ref_emb = np.asarray(embedder.encode(df["Reference Answer"].tolist(), **kw))
    stu_emb = np.asarray(embedder.encode(df["Student Answer"].tolist(), **kw))
    que_emb = np.asarray(embedder.encode(df["Question"].tolist(), **kw))

    df["Semantic Similarity"] = _rowwise_cosine(ref_emb, stu_emb)
    df["question_student_similarity"] = _rowwise_cosine(que_emb, stu_emb)
    ref_len = df["Reference Answer"].str.len()
    df["Length Ratio"] = np.where(ref_len > 0, df["Student Answer"].str.len() / ref_len.replace(0, 1), 0.0)

    stu_pos = df["Student Answer"].map(pos_profile)
    ref_pos = df["Reference Answer"].map(pos_profile)
    for key in POS_KEYS:
        df[f"POS_{key}_diff"] = stu_pos.map(lambda p, k=key: p[k]) - ref_pos.map(lambda p, k=key: p[k])
    df["POS_similarity"] = 1 - df[[f"POS_{k}_diff" for k in POS_KEYS]].abs().sum(axis=1) / len(POS_KEYS)

    df["flesch_kincaid_ratio"] = [
        flesch_kincaid_ratio(r, s) for r, s in zip(df["Reference Answer"], df["Student Answer"])
    ]
    return df
