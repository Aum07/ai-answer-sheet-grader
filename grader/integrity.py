"""Defences against answers that game the grader.

Three layers, all model-agnostic:

1. Cleaning      - drop repeated sentences and sentences unrelated to the question or
                   reference (filler), so padding no longer inflates length or similarity.
2. Coverage      - how many of the reference answer's points the student actually covered,
                   matched sentence by sentence (used as model features).
3. Integrity     - flag answers that copy the question, are a bare keyword list, or answer a
                   different question; flagged answers are capped and marked for teacher review.

Thresholds are calibrated on real answers at fixed false-positive rates by
scripts/calibrate_integrity.py and stored in models/integrity.json (defaults below are used
until then). Off-topic has two tiers: "off_topic" (high confidence, score capped) and
"possibly_off_topic" (flagged for review only, score unchanged).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

from . import config

THRESHOLDS_PATH = config.MODELS_DIR / "integrity.json"
DEFAULT_THRESHOLDS = {
    "relevance_min": 0.20,  # sentence similarity (to question / any reference sentence) below this = filler
    "off_topic_cap": {"abs": 0.10, "margin": 0.30},  # own-reference similarity below abs, or another
    "off_topic_review": {"abs": 0.20, "margin": 0.10},  # question's reference closer by more than margin
    "keyword_min_words": 6,
}
COVERED_MIN = 0.50  # a reference sentence counts as covered if some student sentence reaches this similarity
FLAG_CAP = 2  # max score for a capped answer (out of 10)
CAP_FLAGS = {"copies_question", "keyword_list", "off_topic"}

FLAG_MESSAGES = {
    "copies_question": "the answer mostly restates the question",
    "keyword_list": "the answer is a list of terms rather than sentences",
    "off_topic": "the answer appears to address a different question",
    "possibly_off_topic": "the answer may address a different question",
}


@lru_cache(maxsize=1)
def thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        return {**DEFAULT_THRESHOLDS, **json.loads(THRESHOLDS_PATH.read_text())["thresholds"]}
    return DEFAULT_THRESHOLDS

STOPWORDS = frozenset(
    """a an the and or but if then else of to in on at by for with from as is are was were be been being
    it its this that these those there their they them we you your our he she his her i me my not no yes
    can could should would will may might must do does did done so such than too very also into over under
    about which who whom what when where why how all any each both more most other some only same own just
    has have had having one uses use used""".split()
)

_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_WORD = re.compile(r"[a-z][a-z0-9'-]*")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text or "") if p and p.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])


def _norm(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))


def dedupe(sentences: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for s in sentences:
        key = _norm(s)
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


class SentenceCache:
    """Embeds each distinct sentence once (answers on a sheet share reference sentences)."""

    def __init__(self, embedder, batch_size: int = 64):
        self.embedder, self.batch_size, self.vecs = embedder, batch_size, {}

    def add(self, sentences: Iterable[str]) -> None:
        new = list(dict.fromkeys(s for s in sentences if s not in self.vecs))
        if new:
            emb = np.asarray(self.embedder.encode(new, batch_size=self.batch_size, show_progress_bar=False), dtype=float)
            for s, v in zip(new, _unit(emb)):
                self.vecs[s] = v

    def matrix(self, sentences: Sequence[str]) -> np.ndarray:
        return np.vstack([self.vecs[s] for s in sentences])


def clean_and_cover(questions: Sequence[str], references: Sequence[str], answers: Sequence[str], embedder, clean: bool = True):
    """Return (cleaned answers, coverage_mean, coverage_frac) for aligned lists."""
    stu = [dedupe(split_sentences(a)) if clean else split_sentences(a) for a in answers]
    ref = [split_sentences(r) for r in references]
    cache = SentenceCache(embedder)
    cache.add([s for ss in stu for s in ss] + [s for rs in ref for s in rs] + list(questions))

    cleaned, cov_mean, cov_frac = [], [], []
    for q, rs, ss, original in zip(questions, ref, stu, answers):
        if not ss or not rs:
            cleaned.append(original or "")
            cov_mean.append(0.0)
            cov_frac.append(0.0)
            continue
        S = cache.matrix(ss)
        anchors = cache.matrix(rs + [q])
        relevance = (S @ anchors.T).max(axis=1)
        if clean:
            keep = relevance >= thresholds()["relevance_min"]
            keep[int(relevance.argmax())] = True  # never drop the whole answer
            ss = [s for s, k in zip(ss, keep) if k]
            S = S[keep]
        cleaned.append(" ".join(ss))
        best = (cache.matrix(rs) @ S.T).max(axis=1)  # for each reference point, best student sentence
        cov_mean.append(float(best.mean()))
        cov_frac.append(float((best >= COVERED_MIN).mean()))
    return cleaned, np.array(cov_mean), np.array(cov_frac)


def _content_tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 2]


def copies_question(answer: str, question: str) -> bool:
    ans = _content_tokens(answer)
    if len(ans) < 2:
        return False
    q = set(_content_tokens(question))
    new = [w for w in ans if w not in q]
    return len(new) <= 2 and len(new) / len(ans) <= 0.15


def is_keyword_list(answer: str, min_words: int | None = None) -> bool:
    words = _WORD.findall(answer.lower())
    if len(words) < (min_words or thresholds()["keyword_min_words"]):
        return False
    stop_share = sum(w in STOPWORDS for w in words) / len(words)
    return stop_share < 0.10


def integrity_flags(
    questions: Sequence[str],
    own_references: Sequence[str],
    answers: Sequence[str],
    embedder,
    all_references: Sequence[str] | Sequence[Sequence[str]] | None = None,
) -> list[list[str]]:
    """Flags per answer.

    `all_references` is the reference answers on the sheet, used by the off-topic check: either
    one list shared by every answer, or one list per answer (e.g. each answer's own assignment).
    """
    n = len(answers)
    if all_references is None:
        sheets = [list(own_references)] * n
    elif all_references and not isinstance(all_references[0], str):
        sheets = [list(s) for s in all_references]
    else:
        sheets = [list(all_references)] * n
    texts = list(dict.fromkeys(list(answers) + list(own_references) + [r for s in sheets for r in s]))
    emb = _unit(np.asarray(embedder.encode(texts, show_progress_bar=False), dtype=float))
    vec = dict(zip(texts, emb))
    t = thresholds()

    flags = []
    for q, own, a, sheet in zip(questions, own_references, answers, sheets):
        f = []
        if copies_question(a, q):
            f.append("copies_question")
        if is_keyword_list(a):
            f.append("keyword_list")
        own_sim = float(vec[a] @ vec[own])
        others = [float(vec[a] @ vec[r]) for r in dict.fromkeys(sheet) if r != own]
        margin = max(others) - own_sim if others else -1.0
        if own_sim < t["off_topic_cap"]["abs"] or margin > t["off_topic_cap"]["margin"]:
            f.append("off_topic")
        elif own_sim < t["off_topic_review"]["abs"] or margin > t["off_topic_review"]["margin"]:
            f.append("possibly_off_topic")
        flags.append(f)
    return flags


def apply_caps(scores: np.ndarray, flags: Sequence[list[str]], cap: int = FLAG_CAP) -> np.ndarray:
    """Cap scores of answers with a high-confidence flag; review-only flags leave the score alone."""
    scores = np.asarray(scores).copy()
    for i, f in enumerate(flags):
        if CAP_FLAGS.intersection(f):
            scores[i] = min(scores[i], cap)
    return scores
