"""Entry point for Hugging Face Spaces (Gradio SDK).

The Space runs this file and serves whatever listens on port 7860, so it starts the regular
FastAPI app (web UI + API) instead of a Gradio interface. Locally, use `uvicorn app.main:app`.
"""

import os
import threading

import uvicorn

from app.main import app
from grader.features import get_embedder


def _warm_up() -> None:
    """Load the embedding model in the background so the first visitor doesn't wait for it."""
    try:
        get_embedder()
    except Exception as exc:  # the app still works; the first request will load it instead
        print(f"Warm-up failed: {exc}")


if __name__ == "__main__":
    threading.Thread(target=_warm_up, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
