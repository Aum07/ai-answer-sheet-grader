"""LLM grader: score answers against the reference with an LLM and return written feedback.

All answers to one question are sent in a single request (so a class of 30 costs one call per
question, not 30), with a JSON schema so the reply is always machine-readable. Results are cached
on disk, so re-running an experiment never pays for the same answer twice.

Needs OPENAI_API_KEY in the environment. Model and prices are configurable:
    LLM_MODEL (default gpt-4o-mini), LLM_PRICE_IN / LLM_PRICE_OUT (USD per 1M tokens)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"  # bump when the prompt changes, so cached grades aren't reused
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
PRICE_IN = float(os.getenv("LLM_PRICE_IN", 0.15))  # USD per 1M input tokens (gpt-4o-mini list price)
PRICE_OUT = float(os.getenv("LLM_PRICE_OUT", 0.60))
BATCH_SIZE = 20
CACHE_DIR = config.ROOT_DIR / "data" / "llm_cache"

SYSTEM_PROMPT = """You are an experienced, fair examiner marking short written exam answers.

For each student answer, compare it with the reference answer and give a score from 0 to 10:
- 10: fully correct and complete; covers the key points of the reference
- 7-9: correct with minor omissions or imprecision
- 4-6: partly correct; some key points missing or partly wrong
- 1-3: mostly incorrect or very incomplete, but shows some relevant understanding
- 0: blank, irrelevant, or entirely wrong

Rules:
- Mark the substance. Correct answers worded differently from the reference get full credit;
  correct relevant points not in the reference also count.
- Do NOT reward length, repetition, filler sentences, restating the question, or lists of
  keywords without explanation.
- An answer that addresses a different question scores 0.
- Feedback: one or two short sentences addressed to the student, naming what was good and what
  was missing or wrong. No more than 40 words.
Return a grade for every answer id you are given."""

RESPONSE_SCHEMA = {
    "name": "grades",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "grades": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "score": {"type": "integer"},
                        "feedback": {"type": "string"},
                    },
                    "required": ["id", "score", "feedback"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["grades"],
        "additionalProperties": False,
    },
}


@dataclass
class LLMGrade:
    score: int
    feedback: str


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    failures: list = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * PRICE_IN + self.output_tokens * PRICE_OUT) / 1e6


def _key(question: str, reference: str, answer: str) -> str:
    raw = json.dumps([PROMPT_VERSION, MODEL, question, reference, answer])
    return hashlib.sha256(raw.encode()).hexdigest()


class LLMGrader:
    def __init__(self, client=None, cache_dir: Path | None = CACHE_DIR, max_retries: int = 4):
        if client is None:
            from openai import OpenAI  # imported lazily: only needed when the LLM grader is used

            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("Set the OPENAI_API_KEY environment variable to use the LLM grader")
            client = OpenAI()
        self.client = client
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.usage = Usage()
        self._lock = threading.Lock()  # grade_question may be called from several threads
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- cache -------------------------------------------------------------------------
    def _cached(self, key: str) -> LLMGrade | None:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return LLMGrade(d["score"], d["feedback"])
        return None

    def _store(self, key: str, grade: LLMGrade) -> None:
        if self.cache_dir:
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps({"score": grade.score, "feedback": grade.feedback}), encoding="utf-8"
            )

    # ---- grading -----------------------------------------------------------------------
    def grade_question(self, question: str, reference: str, answers: list[str]) -> list[LLMGrade]:
        """Grade all answers to one question. Returns grades in the same order as `answers`."""
        out: list[LLMGrade | None] = [None] * len(answers)
        todo = []
        for i, a in enumerate(answers):
            if not a.strip():
                out[i] = LLMGrade(0, "No answer was given.")
                continue
            hit = self._cached(_key(question, reference, a))
            if hit:
                out[i] = hit
                with self._lock:
                    self.usage.cached += 1
            else:
                todo.append(i)

        for start in range(0, len(todo), BATCH_SIZE):
            batch = todo[start : start + BATCH_SIZE]
            grades = self._call(question, reference, {str(j + 1): answers[i] for j, i in enumerate(batch)})
            for j, i in enumerate(batch):
                g = grades.get(str(j + 1))
                if g is None:  # model skipped an id even after retries
                    with self._lock:
                        self.usage.failures.append(answers[i][:60])
                    g = LLMGrade(0, "Could not be graded automatically; please review.")
                else:
                    self._store(_key(question, reference, answers[i]), g)
                out[i] = g
        return out  # type: ignore[return-value]

    def _call(self, question: str, reference: str, answers: dict[str, str]) -> dict[str, LLMGrade]:
        user = json.dumps(
            {
                "question": question,
                "reference_answer": reference,
                "student_answers": [{"id": k, "answer": v} for k, v in answers.items()],
            },
            ensure_ascii=False,
        )
        remaining = dict(answers)
        result: dict[str, LLMGrade] = {}
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=MODEL,
                    temperature=0,
                    response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                )
                with self._lock:
                    self.usage.seconds += time.time() - t0
                    self.usage.calls += 1
                    if resp.usage:
                        self.usage.input_tokens += resp.usage.prompt_tokens
                        self.usage.output_tokens += resp.usage.completion_tokens
                for g in json.loads(resp.choices[0].message.content)["grades"]:
                    if g["id"] in remaining:
                        result[g["id"]] = LLMGrade(max(0, min(10, int(g["score"]))), g["feedback"].strip())
                        remaining.pop(g["id"])
                if not remaining:
                    return result
                # ask again only for the ids the model skipped
                user = json.dumps(
                    {
                        "question": question,
                        "reference_answer": reference,
                        "student_answers": [{"id": k, "answer": v} for k, v in remaining.items()],
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:  # rate limits, timeouts, malformed replies
                wait = 2**attempt
                logger.warning("LLM call failed (%s); retrying in %ss", exc, wait)
                time.sleep(wait)
        return result
