"""API tests with the grading pipeline stubbed out (no model download needed)."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app import main as api  # noqa: E402
from grader import config, pipeline  # noqa: E402
from grader.pipeline import QuestionResult, SheetResult  # noqa: E402

PDF = b"%PDF-1.4 fake"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api.scoring, "models_available", lambda: True)
    monkeypatch.setattr(api.scoring, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "parse_reference", lambda path: ["ref"])

    def fake_grade(ref, path, name, embedder=None, reference=None):
        if "broken" in name.lower():
            raise ValueError("unreadable")
        return SheetResult(name, [QuestionResult("1", "Q?", "A.", 8, 10, "Good.")])

    monkeypatch.setattr(pipeline, "grade_sheet", fake_grade)
    api.rate_limit.hits.clear()
    with TestClient(api.app) as c:
        yield c


def post(client, students, headers=None):
    files = [("reference_pdf", ("ref.pdf", PDF, "application/pdf"))]
    files += [("student_pdfs", (name, data, "application/pdf")) for name, data in students]
    return client.post("/evaluate/", files=files, headers=headers or {})


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_root_redirects_to_ui(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/static/index.html"


def test_evaluate_and_download(client):
    r = post(client, [("student_01.pdf", PDF), ("student_02.pdf", PDF)])
    assert r.status_code == 200, r.text
    body = r.json()
    assert [e["student_name"] for e in body["evaluations"]] == ["Student 01", "Student 02"]
    assert body["evaluations"][0]["percentage"] == 80.0

    pdf = client.get(f"/download/{body['request_id']}", params={"student": "Student 02"})
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    z = client.get(f"/download-multiple/{body['request_id']}")
    assert z.status_code == 200 and z.headers["content-type"] == "application/zip"


def test_one_bad_sheet_does_not_fail_the_class(client):
    r = post(client, [("good.pdf", PDF), ("broken.pdf", PDF), ("fake.pdf", b"not a pdf")])
    body = r.json()
    assert r.status_code == 200
    assert [e["student_name"] for e in body["evaluations"]] == ["Good"]
    assert {f["student_name"] for f in body["failed"]} == {"Broken", "Fake"}


def test_rejects_non_pdf_reference(client):
    files = [("reference_pdf", ("ref.txt", b"hello", "text/plain")), ("student_pdfs", ("s.pdf", PDF, "application/pdf"))]
    assert client.post("/evaluate/", files=files).status_code == 400


def test_unknown_request_id_is_404(client):
    assert client.get("/download/nope").status_code == 404


def test_api_key_enforced_when_set(client, monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "secret")
    assert post(client, [("s.pdf", PDF)]).status_code == 401
    assert post(client, [("s.pdf", PDF)], headers={"X-API-Key": "secret"}).status_code == 200


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(api.rate_limit, "per_minute", 2)
    codes = [post(client, [("s.pdf", PDF)]).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_demo_endpoint(client, monkeypatch):
    api._demo_cache.clear()
    monkeypatch.setattr(api, "_grade_demo", lambda: {"evaluations": [SheetResult("Sample 1 - High Performer", [QuestionResult("1", "Q?", "A.", 9, 10, "Excellent.")]).to_dict()], "failed": []})
    r = client.post("/demo/")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["evaluations"][0]["student_name"].startswith("Sample 1")
    assert client.get(f"/download/{body['request_id']}").status_code == 200
