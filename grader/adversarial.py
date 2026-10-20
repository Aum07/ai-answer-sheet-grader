"""Answer perturbations used to probe whether the grader rewards substance or surface form."""

from __future__ import annotations

import re

FILLER = (
    " This is a very important topic that has been widely discussed and studied over many years."
    " There are many different aspects to consider, and each of them plays a significant role in"
    " understanding the overall concept. In practice, it is essential to think carefully about these"
    " ideas and how they apply in different situations. In conclusion, this concept is clearly very"
    " significant and should be understood well."
)

_STOP = set(
    """a an the and or but if then else of to in on at by for with from as is are was were be been being
    it its this that these those there their they them we you your our he she his her i me my not no yes
    can could should would will may might must do does did done so such than too very also into over under
    about which who whom what when where why how all any each both more most other some only same own just""".split()
)


def keywords(reference: str, n: int = 15) -> list[str]:
    """Content words of the reference answer, in order of first appearance."""
    seen, out = set(), []
    for w in re.findall(r"[A-Za-z][A-Za-z-]+", reference.lower()):
        if len(w) > 3 and w not in _STOP and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:n]


def pad(answer: str, **_) -> str:
    return answer.rstrip() + FILLER


def repeat(answer: str, **_) -> str:
    return f"{answer.strip()} {answer.strip()}"


def copy_question(answer: str, question: str, **_) -> str:
    return question


def keyword_stuff(answer: str, reference: str, **_) -> str:
    return " ".join(keywords(reference))


def truncate(answer: str, **_) -> str:
    words = answer.split()
    return " ".join(words[: max(1, len(words) // 2)])


def copy_reference(answer: str, reference: str, **_) -> str:
    return reference


# name -> (function, expected effect on score, short description)
ATTACKS = {
    "padding": (pad, "no increase", "append ~70 words of fluent, content-free filler"),
    "repetition": (repeat, "no increase", "the same answer written twice"),
    "copy_question": (copy_question, "decrease", "answer replaced by the question text"),
    "keyword_stuffing": (keyword_stuff, "no increase", "15 key terms from the reference, no sentences"),
    "truncation": (truncate, "decrease", "only the first half of the answer"),
    "off_topic": (None, "decrease", "a real answer to a different question"),  # needs other rows; see script
    "copy_reference": (copy_reference, "increase", "control: the model answer itself"),
}
