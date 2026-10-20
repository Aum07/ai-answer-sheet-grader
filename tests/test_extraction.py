import pytest

from grader.extraction import NO_ANSWER, QA, align_answers, normalize_text, parse_questions_and_answers


def test_normalize_repairs_broken_ligatures():
    # Real output from a Word-exported sample sheet
    assert normalize_text("QuesƟon 1 the diﬀerences, theŌ, seƩled") == "Question 1 the differences, theft, settled"
    # pdfplumber/pdfminer emits glyph ids for the same ligatures
    assert normalize_text("Ques(cid:415)on 2 o(cid:332)en be(cid:425)er se(cid:427)ngs") == "Question 2 often better settings"


@pytest.mark.parametrize(
    "text",
    [
        "Question 1: What is AI?\nAnswer: Machines that think.\nQuestion 2: What is ML?\nAnswer: Learning from data.",
        "Question 1\nWhat is AI?\nAnswer:\nMachines that think.\nQuestion 2\nWhat is ML?\nAnswer:\nLearning from data.",
        # pypdf sometimes glues lines together
        "Student 5: Good StudentQuestion 1: What is AI?Answer: Machines that think.Question 2: What is ML?Answer: Learning from data.",
        "Q1. What is AI? Ans: Machines that think. Q2) What is ML? Ans - Learning from data.",
        "QuesƟon 1: What is AI?\nAnswer: Machines that think.\nQuesƟon 2: What is ML?\nAnswer: Learning from data.",
    ],
)
def test_parse_supports_common_layouts(text):
    qas = parse_questions_and_answers(text)
    assert [q.id for q in qas] == ["1", "2"]
    assert qas[0].question == "What is AI?"
    assert qas[0].answer == "Machines that think."
    assert qas[1].answer == "Learning from data."


def test_header_before_first_question_is_ignored():
    qas = parse_questions_and_answers("Law Answer Sheet - Student 9\nQuestion 1: Define tort.\nAnswer: A civil wrong.")
    assert len(qas) == 1 and qas[0].question == "Define tort."


def test_missing_answer_is_marked():
    qas = parse_questions_and_answers("Question 1: Define tort.\nQuestion 2: Define crime.\nAnswer: An offence.")
    assert qas[0].answer == NO_ANSWER
    assert qas[1].answer == "An offence."


def test_no_questions_returns_empty():
    assert parse_questions_and_answers("just some text") == []


def test_word_iq_is_not_a_question_marker():
    qas = parse_questions_and_answers("Question 1: What is IQ?\nAnswer: An IQ 5 points higher is not much.")
    assert len(qas) == 1


def test_align_by_id_then_by_question_text():
    ref = [QA("1", "What is supervised learning?", "r1"), QA("2", "What is reinforcement learning?", "r2")]
    stu = [QA("7", "What is reinforcement learning?", "s2"), QA("1", "What is supervised learning?", "s1")]
    rows = align_answers(ref, stu)
    assert [r["Student Answer"] for r in rows] == ["s1", "s2"]


def test_align_unmatched_question_gets_no_answer():
    rows = align_answers([QA("1", "Explain photosynthesis.", "r")], [QA("2", "Name three rivers.", "s")])
    assert rows[0]["Student Answer"] == NO_ANSWER


def test_extract_text_from_generated_pdf(pdf_factory):
    pytest.importorskip("fitz")
    from grader.extraction import extract_text_from_pdf

    pdf = pdf_factory("ref.pdf", ["Question 1: What is AI?", "Answer: Machines that think."])
    qas = parse_questions_and_answers(extract_text_from_pdf(pdf))
    assert qas == [QA("1", "What is AI?", "Machines that think.")]
