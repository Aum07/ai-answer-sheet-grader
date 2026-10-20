"""PDF and ZIP report generation."""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "student"


def _p(text: str, style) -> Paragraph:
    # Paragraph parses a mini-HTML dialect: escape student text so "<" or "&" can't break it.
    return Paragraph(escape(str(text)), style)


def student_report_pdf(student: dict) -> bytes:
    """Render one student's evaluation (the dict from SheetResult.to_dict()) as PDF bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=f"Evaluation - {student['student_name']}")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCenter", parent=styles["Heading1"], fontSize=18, spaceAfter=24, alignment=1)
    cell = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=9, leading=12)

    story = [Paragraph("Answer Sheet Evaluation Report", title)]
    info = Table(
        [
            ["Student", _p(student["student_name"], cell)],
            ["Total score", f"{student['total_score']} / {student['max_possible_score']}"],
            ["Percentage", f"{student['percentage']}%"],
            ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ],
        colWidths=[1.6 * inch, 4.4 * inch],
    )
    info.setStyle(_table_style(colors.lightgrey))
    story += [info, Spacer(1, 18), Paragraph("Question-wise breakdown", styles["Heading2"])]

    for r in student["results"]:
        story.append(_p(f"Question {r['question_id']}: {r['question']}", styles["Heading4"]))
        story.append(_p(r["student_answer"], styles["Normal"]))
        story.append(Spacer(1, 6))
        t = Table(
            [["Score", f"{r['score']} / {r['max_score']}" + ("  (flagged for review)" if r.get("flags") else "")],
             ["Feedback", _p(r["feedback"], cell)]],
            colWidths=[1.2 * inch, 4.8 * inch],
        )
        t.setStyle(_table_style(colors.lightblue))
        story += [t, Spacer(1, 14)]

    story.append(
        _p(
            "Scores are machine-generated estimates and should be reviewed by a teacher before use.",
            ParagraphStyle("Note", parent=styles["Italic"], fontSize=8, textColor=colors.grey),
        )
    )
    doc.build(story)
    return buf.getvalue()


def _table_style(label_bg) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (0, -1), label_bg),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
    )


def class_summary_text(students: list[dict]) -> str:
    pct = [s["percentage"] for s in students]
    lines = [
        "CLASS EVALUATION SUMMARY",
        "=" * 50,
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"Students: {len(students)}",
    ]
    if pct:
        lines += [
            f"Average: {sum(pct) / len(pct):.2f}%",
            f"Highest: {max(pct):.2f}%",
            f"Lowest:  {min(pct):.2f}%",
        ]
    lines += ["", "RANKING", "-" * 50]
    for i, s in enumerate(sorted(students, key=lambda s: s["percentage"], reverse=True), 1):
        lines.append(
            f"{i:2d}. {s['student_name']:<25} {s['percentage']:6.2f}%  ({s['total_score']}/{s['max_possible_score']})"
        )
    return "\n".join(lines) + "\n"


def class_reports_zip(students: list[dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        used: set[str] = set()
        for s in students:
            name = safe_filename(s["student_name"])
            fname, n = f"{name}_report.pdf", 2
            while fname in used:  # two students with the same name
                fname, n = f"{name}_{n}_report.pdf", n + 1
            used.add(fname)
            zf.writestr(fname, student_report_pdf(s))
        zf.writestr("class_summary.txt", class_summary_text(students))
    return buf.getvalue()
