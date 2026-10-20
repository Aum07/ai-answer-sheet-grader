"""End-to-end grading: reference PDF + student PDF -> per-question scores."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from . import config
from .extraction import NO_ANSWER, align_answers, extract_text_from_pdf, parse_questions_and_answers
from .features import Embedder, compute_features, get_embedder
from .feedback import feedback_for, feedback_for_flags
from .integrity import apply_caps, integrity_flags
from .scoring import predict_scores

logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    question_id: str
    question: str
    student_answer: str
    score: int
    max_score: int
    feedback: str
    flags: list[str] = field(default_factory=list)


@dataclass
class SheetResult:
    student_name: str
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def total_score(self) -> int:
        return sum(r.score for r in self.results)

    @property
    def max_possible_score(self) -> int:
        return sum(r.max_score for r in self.results)

    @property
    def percentage(self) -> float:
        return round(100 * self.total_score / self.max_possible_score, 2) if self.max_possible_score else 0.0

    def to_dict(self) -> dict:
        return {
            "student_name": self.student_name,
            "results": [asdict(r) for r in self.results],
            "total_score": self.total_score,
            "max_possible_score": self.max_possible_score,
            "percentage": self.percentage,
        }


def grade_frame(
    pairs: pd.DataFrame,
    embedder: Embedder | None = None,
    model=None,
    scaler=None,
    defences: bool = True,
    all_references: list[str] | None = None,
) -> pd.DataFrame:
    """Score a frame of (ID, Question, Reference Answer, Student Answer) rows.

    Blank answers get 0 without going through the model. With `defences`, answers are cleaned
    (repeats and filler removed) before scoring, and integrity checks cap and flag answers that
    copy the question, are keyword lists, or address a different question. `all_references`
    (every reference answer on the sheet) sharpens the off-topic check.
    """
    pairs = pairs.copy()
    blank = pairs["Student Answer"].str.strip().isin(["", NO_ANSWER])
    pairs["Predicted Score"] = 0
    all_flags: list[list[str]] = [[] for _ in range(len(pairs))]
    answered = pairs[~blank]
    if not answered.empty:
        embedder = embedder or get_embedder()
        feats = compute_features(answered, embedder=embedder, clean=defences)
        scores = predict_scores(feats, model=model, scaler=scaler)
        if defences:
            flags = integrity_flags(
                answered["Question"].tolist(),
                answered["Reference Answer"].tolist(),
                answered["Student Answer"].tolist(),
                embedder,
                all_references=all_references or pairs["Reference Answer"].tolist(),
            )
            scores = apply_caps(scores, flags)
            for pos, f in zip((~blank).to_numpy().nonzero()[0], flags):
                all_flags[pos] = f
        pairs.loc[~blank, "Predicted Score"] = scores
    pairs["Flags"] = all_flags
    return pairs


def parse_reference(reference_pdf: str | Path) -> list:
    reference = parse_questions_and_answers(extract_text_from_pdf(reference_pdf))
    if not reference:
        raise ValueError(
            "No questions found in the reference sheet. Each question must start with "
            "'Question N' and its model answer with 'Answer:'."
        )
    return reference


def grade_sheet(
    reference_pdf: str | Path,
    student_pdf: str | Path,
    student_name: str,
    embedder: Embedder | None = None,
    reference: list | None = None,
) -> SheetResult:
    """Grade one student sheet. Pass a pre-parsed `reference` to avoid re-reading it per student."""
    reference = reference if reference is not None else parse_reference(reference_pdf)
    student = parse_questions_and_answers(extract_text_from_pdf(student_pdf))
    pairs = pd.DataFrame(align_answers(reference, student))
    scored = grade_frame(pairs, embedder=embedder)

    result = SheetResult(student_name=student_name)
    for qid, question, answer, score, flags in zip(
        scored["ID"], scored["Question"], scored["Student Answer"], scored["Predicted Score"], scored["Flags"]
    ):
        score = int(score)
        result.results.append(
            QuestionResult(
                question_id=str(qid),
                question=question,
                student_answer=answer,
                score=score,
                max_score=config.MAX_SCORE,
                feedback=feedback_for_flags(flags, score) if flags else feedback_for(score),
                flags=list(flags),
            )
        )
    logger.info("Graded %s: %d/%d", student_name, result.total_score, result.max_possible_score)
    return result
