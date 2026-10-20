import io
import zipfile

from grader.reports import class_reports_zip, safe_filename, student_report_pdf


def student(name="Student 01", answer="An answer with <script>alert(1)</script> & symbols"):
    return {
        "student_name": name,
        "results": [
            {"question_id": "1", "question": "Q?", "student_answer": answer, "score": 7, "max_score": 10, "feedback": "Good."}
        ],
        "total_score": 7,
        "max_possible_score": 10,
        "percentage": 70.0,
    }


def test_pdf_renders_with_markup_in_answers():
    pdf = student_report_pdf(student())
    assert pdf.startswith(b"%PDF")


def test_zip_has_unique_names_and_summary():
    data = class_reports_zip([student("Sam Lee"), student("Sam Lee"), student("A/B")])
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert len(set(names)) == len(names) == 4
    assert "class_summary.txt" in names
    assert "A_B_report.pdf" in names


def test_safe_filename():
    assert safe_filename("../../etc passwd") == "etc_passwd"
    assert safe_filename("") == "student"
