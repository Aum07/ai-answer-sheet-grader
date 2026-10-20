"""FastAPI service: upload a reference sheet + student sheets, get scores and PDF reports."""

# No `from __future__ import annotations` here: FastAPI must see real types on
# dependency signatures (e.g. RateLimiter.__call__(request: Request)).
import logging
import tempfile
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from grader import __version__, config, pipeline, reports, scoring

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("grader.api")

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = int(config.MAX_UPLOAD_MB * 1024 * 1024)


# ---------- response models ----------
class QuestionResultOut(BaseModel):
    question_id: str
    question: str
    student_answer: str
    score: int
    max_score: int
    feedback: str
    flags: List[str] = []


class StudentEvaluation(BaseModel):
    student_name: str
    results: List[QuestionResultOut]
    total_score: int
    max_possible_score: int
    percentage: float


class FailedSheet(BaseModel):
    student_name: str
    error: str


class EvaluationResponse(BaseModel):
    request_id: str
    evaluations: List[StudentEvaluation]
    failed: List[FailedSheet] = []
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    version: str


# ---------- in-memory result store (for report downloads) ----------
class ResultStore:
    """Keeps recent evaluations so the UI can download reports. Single-process only."""

    def __init__(self, ttl: int):
        self.ttl = ttl
        self._items: dict[str, tuple[float, dict]] = {}

    def put(self, key: str, value: dict) -> None:
        self._purge()
        self._items[key] = (time.time(), value)

    def get(self, key: str) -> Optional[dict]:
        self._purge()
        item = self._items.get(key)
        return item[1] if item else None

    def _purge(self) -> None:
        cutoff = time.time() - self.ttl
        for k in [k for k, (t, _) in self._items.items() if t < cutoff]:
            del self._items[k]


store = ResultStore(config.RESULT_TTL_SECONDS)


# ---------- dependencies ----------
class RateLimiter:
    """Sliding-window limit per client IP on the expensive /evaluate endpoint."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    def __call__(self, request: Request) -> None:
        if self.per_minute <= 0:
            return
        ip = request.client.host if request.client else "unknown"
        now, window = time.time(), self.hits[ip]
        while window and window[0] < now - 60:
            window.popleft()
        if len(window) >= self.per_minute:
            raise HTTPException(429, "Too many requests, please wait a minute.")
        window.append(now)


rate_limit = RateLimiter(config.RATE_LIMIT_PER_MINUTE)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Depends(api_key_header)) -> None:
    if config.API_KEY and api_key != config.API_KEY:
        raise HTTPException(401, "Invalid or missing API key", headers={"WWW-Authenticate": "ApiKey"})


# ---------- app ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if scoring.models_available():
        scoring.load_model()
        logger.info("Model %r loaded from %s", config.MODEL_NAME, config.MODEL_PATH)
    else:
        logger.error("Model files missing: %s / %s", config.MODEL_PATH, config.SCALER_PATH)
    yield


app = FastAPI(
    title="AI Answer Sheet Grader",
    description="Scores free-text exam answers against a reference answer sheet.",
    version=__version__,
    lifespan=lifespan,
)
if config.CORS_ORIGINS:  # same-origin UI needs no CORS; only enable for explicit origins
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def no_cache_for_ui(request: Request, call_next):
    """Make browsers revalidate UI files, so an updated results.js is never served stale."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def student_name_from_filename(filename: Optional[str]) -> str:
    stem = Path(filename or "").stem
    name = " ".join(w.capitalize() for w in stem.replace("_", " ").replace("-", " ").split())
    return name or "Unknown Student"


async def save_pdf(upload: UploadFile, workdir: Path) -> Path:
    """Validate an upload (extension, size, PDF magic bytes) and save it into workdir."""
    if not (upload.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, f"{upload.filename}: only PDF files are accepted")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{upload.filename}: larger than {config.MAX_UPLOAD_MB:g} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, f"{upload.filename}: not a valid PDF")
    path = workdir / f"{uuid.uuid4().hex}.pdf"
    path.write_bytes(data)
    return path


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/static/index.html")


@app.get("/health", response_model=HealthResponse)
def health():
    ok = scoring.models_available()
    return {"status": "ok" if ok else "degraded", "model_loaded": ok, "model_name": config.MODEL_NAME, "version": __version__}


@app.post(
    "/evaluate/",
    response_model=EvaluationResponse,
    dependencies=[Depends(rate_limit), Depends(verify_api_key)],
)
async def evaluate(reference_pdf: UploadFile = File(...), student_pdfs: List[UploadFile] = File(...)):
    if not scoring.models_available():
        raise HTTPException(503, "Model files are missing on the server")
    if len(student_pdfs) > config.MAX_STUDENT_FILES:
        raise HTTPException(400, f"At most {config.MAX_STUDENT_FILES} student sheets per request")

    request_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="grader_") as tmp:
        workdir = Path(tmp)
        ref_path = await save_pdf(reference_pdf, workdir)
        try:
            reference = await run_in_threadpool(pipeline.parse_reference, ref_path)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

        evaluations, failed = [], []
        for upload in student_pdfs:
            name = student_name_from_filename(upload.filename)
            try:
                path = await save_pdf(upload, workdir)
                # CPU-heavy (embeddings, POS tagging): keep it off the event loop.
                result = await run_in_threadpool(pipeline.grade_sheet, ref_path, path, name, None, reference)
                evaluations.append(result.to_dict())
            except HTTPException as exc:
                failed.append({"student_name": name, "error": exc.detail})
            except Exception as exc:  # one bad sheet must not fail the whole class
                logger.exception("Failed to grade %s", name)
                failed.append({"student_name": name, "error": f"Could not grade this sheet: {exc}"})

    if not evaluations:
        detail = "; ".join(f"{f['student_name']}: {f['error']}" for f in failed) or "No sheets graded"
        raise HTTPException(422, detail)

    response = {
        "request_id": request_id,
        "evaluations": evaluations,
        "failed": failed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    store.put(request_id, response)
    logger.info("Request %s: graded %d sheet(s), %d failed", request_id, len(evaluations), len(failed))
    return response


SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples" / "ai"
# Six anonymised sample sheets spanning the intended quality range (see samples/manifest.csv)
DEMO_SHEETS = [
    ("student_29.pdf", "High Performer"),
    ("student_08.pdf", "Excellent with Some Gaps"),
    ("student_14.pdf", "Average"),
    ("student_04.pdf", "Below Average"),
    ("student_11.pdf", "Struggling with Concepts"),
    ("student_10.pdf", "Poor"),
]
_demo_cache: dict = {}


def _grade_demo() -> dict:
    ref = SAMPLES_DIR / "reference.pdf"
    reference = pipeline.parse_reference(ref)
    evaluations = []
    for i, (fname, label) in enumerate(DEMO_SHEETS, 1):
        result = pipeline.grade_sheet(ref, SAMPLES_DIR / fname, f"Sample {i} - {label}", None, reference)
        evaluations.append(result.to_dict())
    return {"evaluations": evaluations, "failed": []}


@app.post("/demo/", response_model=EvaluationResponse, dependencies=[Depends(rate_limit)])
async def demo():
    """Grade six built-in sample sheets (AI subject) so visitors can try the app without PDFs."""
    if not (SAMPLES_DIR / "reference.pdf").exists():
        raise HTTPException(404, "Sample sheets are not available on this server")
    if "result" not in _demo_cache:  # deterministic, so compute once per process
        _demo_cache["result"] = await run_in_threadpool(_grade_demo)
    response = {
        "request_id": uuid.uuid4().hex,
        **_demo_cache["result"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    store.put(response["request_id"], response)
    return response


def _get_evaluations(request_id: str) -> list[dict]:
    data = store.get(request_id)
    if not data:
        raise HTTPException(404, "Evaluation not found or expired, please run it again")
    return data["evaluations"]


@app.get("/download/{request_id}")
def download_student_report(request_id: str, student: Optional[str] = None):
    evaluations = _get_evaluations(request_id)
    if student:
        match = [e for e in evaluations if e["student_name"].lower() == student.lower()]
        if not match:
            raise HTTPException(404, f"Student '{student}' not found in this evaluation")
        chosen = match[0]
    else:
        chosen = evaluations[0]
    pdf = reports.student_report_pdf(chosen)
    filename = f"{reports.safe_filename(chosen['student_name'])}_report.pdf"
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/download-multiple/{request_id}")
def download_all_reports(request_id: str):
    data = reports.class_reports_zip(_get_evaluations(request_id))
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="reports_{request_id[:8]}.zip"'},
    )


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
