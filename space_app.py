"""Entry point for Hugging Face Spaces (Gradio SDK).

The Space runs this file and serves whatever listens on port 7860, so it starts the regular
FastAPI app (web UI + API) instead of a Gradio interface. Locally, use `uvicorn app.main:app`.
"""

# On Hugging Face ZeroGPU hardware, `spaces` must be imported before torch, and the Space must
# declare at least one @spaces.GPU function. The grader itself runs on CPU; this check just
# satisfies that requirement. Locally (no `spaces` package) it's skipped.
try:
    import spaces
except ImportError:
    spaces = None

import os  # noqa: E402
import threading  # noqa: E402

import uvicorn  # noqa: E402

from app.main import app  # noqa: E402
from grader.features import get_embedder  # noqa: E402

if spaces is not None:

    @spaces.GPU
    def gpu_available() -> bool:
        """Reports whether a GPU is attached (required by ZeroGPU; not used for grading)."""
        import torch

        return torch.cuda.is_available()


def _warm_up() -> None:
    """Load the embedding model in the background so the first visitor doesn't wait for it."""
    try:
        get_embedder()
    except Exception as exc:  # the app still works; the first request will load it instead
        print(f"Warm-up failed: {exc}")


if __name__ == "__main__":
    threading.Thread(target=_warm_up, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
