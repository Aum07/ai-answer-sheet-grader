"""Short, score-banded feedback.

This is deliberately simple: it explains the band a score falls in, it does not
critique the content of the answer.
"""

BANDS = [
    (9, "Excellent answer that demonstrates comprehensive understanding."),
    (7, "Good answer with minor areas for improvement."),
    (5, "Satisfactory answer but needs more detail or clarity."),
    (3, "Answer needs significant improvement in content and structure."),
    (1, "Answer is incomplete or shows limited understanding of the topic."),
    (0, "No answer was found for this question."),
]


def feedback_for_flags(flags: list[str], score: int | None = None) -> str:
    from .integrity import CAP_FLAGS, FLAG_CAP, FLAG_MESSAGES

    reasons = "; ".join(FLAG_MESSAGES.get(f, f) for f in flags)
    if CAP_FLAGS.intersection(flags):
        return f"Flagged for teacher review: {reasons}. Score capped at {FLAG_CAP}/10 until reviewed."
    base = feedback_for(score) + " " if score is not None else ""
    return f"{base}Flagged for teacher review: {reasons}."


def feedback_for(score: int) -> str:
    for threshold, message in BANDS:
        if score >= threshold:
            return message
    return BANDS[-1][1]
