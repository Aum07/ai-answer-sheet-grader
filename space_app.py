"""Hugging Face Spaces entry point: a Gradio front end for the grader.

The free ZeroGPU hardware requires a Gradio app with at least one @spaces.GPU function. The
grader itself runs on CPU; the full FastAPI app (custom web UI + REST API) is for local use
and Docker: `uvicorn app.main:app`.
"""

# `spaces` (ZeroGPU) must be imported before torch. It only exists on Hugging Face.
try:
    import spaces
except ImportError:
    spaces = None

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from grader import config, pipeline, reports  # noqa: E402
from grader.features import get_embedder  # noqa: E402

REPO_URL = "https://github.com/Aum07/ai-answer-sheet-grader"
SAMPLES = Path(__file__).parent / "samples" / "ai"
DEMO_SHEETS = [
    ("student_29.pdf", "High Performer"),
    ("student_08.pdf", "Excellent with Some Gaps"),
    ("student_14.pdf", "Average"),
    ("student_04.pdf", "Below Average"),
    ("student_11.pdf", "Struggling with Concepts"),
    ("student_10.pdf", "Poor"),
]

if spaces is not None:

    @spaces.GPU
    def gpu_available() -> bool:
        """Required by ZeroGPU; grading doesn't use the GPU."""
        import torch

        return torch.cuda.is_available()


def _name(path: str) -> str:
    stem = Path(path).stem
    return " ".join(w.capitalize() for w in stem.replace("_", " ").replace("-", " ").split()) or "Student"


def _grade(reference_pdf: str, students: list[tuple[str, str]]):
    """Grade (name, path) pairs; return summary table, per-question table and a ZIP of reports."""
    reference = pipeline.parse_reference(reference_pdf)
    results = [pipeline.grade_sheet(reference_pdf, path, name, None, reference).to_dict() for name, path in students]

    summary = pd.DataFrame(
        [
            {
                "Student": r["student_name"],
                "Score": f"{r['total_score']} / {r['max_possible_score']}",
                "Percentage": r["percentage"],
                "Flagged answers": sum(bool(q["flags"]) for q in r["results"]),
            }
            for r in results
        ]
    )
    details = pd.DataFrame(
        [
            {
                "Student": r["student_name"],
                "Q": q["question_id"],
                "Score": q["score"],
                "Feedback": q["feedback"],
                "Answer (start)": q["student_answer"][:160] + ("…" if len(q["student_answer"]) > 160 else ""),
            }
            for r in results
            for q in r["results"]
        ]
    )
    zip_path = Path(tempfile.mkdtemp()) / "evaluation_reports.zip"
    zip_path.write_bytes(reports.class_reports_zip(results))
    return summary, details, str(zip_path)


def run_demo():
    return _grade(str(SAMPLES / "reference.pdf"), [(f"Sample {i} - {label}", str(SAMPLES / f)) for i, (f, label) in enumerate(DEMO_SHEETS, 1)])


def run_upload(reference, students):
    if not reference or not students:
        raise gr.Error("Upload a reference sheet and at least one student sheet.")
    if len(students) > config.MAX_STUDENT_FILES:
        raise gr.Error(f"At most {config.MAX_STUDENT_FILES} student sheets at a time.")
    try:
        return _grade(reference, [(_name(p), p) for p in students])
    except ValueError as exc:  # e.g. no "Question N" markers in the reference
        raise gr.Error(str(exc)) from exc


with gr.Blocks(title="AI Answer Sheet Grader", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""# 📝 AI Answer Sheet Grader
Scores free-text exam answers (0–10) against a teacher's reference sheet, with feedback and flags for
answers that look like attempts to game the grader. Trained and evaluated on real, human-marked answers.
**[Code, results and method on GitHub]({REPO_URL})**

*Scores are machine estimates for assisting a marker, not final grades. The first run takes ~30 s while the model loads.*"""
    )
    with gr.Tab("Try 6 sample sheets"):
        gr.Markdown("Anonymised sample answers to 5 AI/ML questions, written at different quality levels.")
        demo_btn = gr.Button("Grade the sample sheets", variant="primary")
    with gr.Tab("Upload your own"):
        gr.Markdown("Sheets must label questions `Question 1`, `Question 2`, … and answers `Answer:`.")
        ref_in = gr.File(label="Reference answer sheet (PDF)", file_types=[".pdf"])
        stu_in = gr.File(label="Student answer sheets (PDFs)", file_types=[".pdf"], file_count="multiple")
        upload_btn = gr.Button("Grade uploaded sheets", variant="primary")

    summary_out = gr.Dataframe(label="Class summary", interactive=False)
    details_out = gr.Dataframe(label="Per-question scores and feedback", interactive=False, wrap=True)
    zip_out = gr.File(label="Download PDF reports (ZIP)")

    demo_btn.click(run_demo, outputs=[summary_out, details_out, zip_out])
    upload_btn.click(run_upload, inputs=[ref_in, stu_in], outputs=[summary_out, details_out, zip_out])


if __name__ == "__main__":
    try:
        get_embedder()  # load the model before the first visitor arrives
    except Exception as exc:
        print(f"Warm-up failed (will load on first request): {exc}")
    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=7860)
