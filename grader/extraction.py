"""PDF text extraction and question/answer parsing."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger(__name__)

NO_ANSWER = "No answer provided"

# Some PDF exporters (e.g. Word with Calibri) map ligatures to odd code points, so
# "Question" comes out as "QuesƟon" and "often" as "oŌen". NFKC fixes the standard
# ligatures (ﬁ, ﬀ, ...); these cover the non-standard ones seen in real sheets.
_LIGATURE_FIXES = {
    # Same broken Calibri ligatures as they appear via PyMuPDF/pypdf ...
    "Ɵ": "ti",  # Ɵ
    "Ō": "ft",  # Ō
    "Ʃ": "tt",  # Ʃ
    "ƫ": "tti",  # ƫ
    # ... and via pdfminer/pdfplumber, which emits raw glyph ids
    "(cid:415)": "ti",
    "(cid:332)": "ft",
    "(cid:425)": "tt",
    "(cid:427)": "tti",
    "": "-",  # Symbol-font bullet
    "→": "->",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
}

# "Question 1:", "Question 1", "Q1.", "Q 2)", "Question 3 -" at the start of a line
# or glued to the previous text (pypdf sometimes drops newlines).
_QUESTION_RE = re.compile(r"(?:Question|\bQ)\s*(\d{1,3})\s*[:.)\-]?\s+", re.IGNORECASE)
_ANSWER_RE = re.compile(r"\bAns(?:wer)?\s*[:.\-]\s*", re.IGNORECASE)


@dataclass
class QA:
    id: str
    question: str
    answer: str


def normalize_text(text: str) -> str:
    """Repair ligatures / smart punctuation and collapse whitespace runs."""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _LIGATURE_FIXES.items():
        text = text.replace(bad, good)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: str | Path, ocr_dpi: int = 300) -> str:
    """Extract text from every page, falling back to Tesseract OCR for image-only pages."""
    import fitz  # PyMuPDF; imported lazily so the parser can be used without it

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:  # corrupt / not a PDF
        raise ValueError(f"Could not open PDF {Path(pdf_path).name}: {exc}") from exc

    pages = []
    with doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text.strip():
                logger.info("OCR on page %d of %s", page_num, Path(pdf_path).name)
                text = _ocr_page(page, ocr_dpi)
            pages.append(text)
    return normalize_text("\n".join(pages))


def _ocr_page(page, dpi: int) -> str:
    import pytesseract
    from PIL import Image

    # Default get_pixmap() renders at 72 dpi, which is far too low for OCR.
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img)


def parse_questions_and_answers(text: str) -> list[QA]:
    """Split sheet text into numbered questions and their answers.

    Expected layout (flexible on punctuation, case and line breaks)::

        Question 1: <question text>
        Answer: <answer text>
        Question 2: ...

    Text before the first question (titles, student headers) is ignored. If the same
    question number appears twice, the first occurrence wins.
    """
    text = normalize_text(text)
    matches = list(_QUESTION_RE.finditer(text))
    items: dict[str, QA] = {}
    for i, m in enumerate(matches):
        qid = str(int(m.group(1)))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.end() : end].strip()
        parts = _ANSWER_RE.split(block, maxsplit=1)
        question = _squash(parts[0])
        answer = _squash(parts[1]) if len(parts) > 1 else ""
        if qid not in items:
            items[qid] = QA(qid, question, answer or NO_ANSWER)
    if not items:
        logger.warning("No 'Question N' markers found in sheet text")
    return list(items.values())


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def align_answers(reference: list[QA], student: list[QA], min_similarity: float = 0.8) -> list[dict]:
    """Pair each reference question with the student's answer.

    Match by question number first; if the number is missing, fall back to the most
    similar question text (students sometimes renumber or skip questions).
    """
    by_id = {qa.id: qa for qa in student}
    rows = []
    for ref in reference:
        stu = by_id.get(ref.id)
        if stu is None and student:
            best = max(student, key=lambda s: SequenceMatcher(None, ref.question, s.question).ratio())
            if SequenceMatcher(None, ref.question, best.question).ratio() >= min_similarity:
                stu = best
        rows.append(
            {
                "ID": ref.id,
                "Question": ref.question,
                "Reference Answer": ref.answer,
                "Student Answer": stu.answer if stu else NO_ANSWER,
            }
        )
    return rows
