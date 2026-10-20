"""Hugging Face Spaces entry point.

Serves the project's own website and API (app/main.py) at "/". Free ZeroGPU hardware expects a
Gradio app with a @spaces.GPU function, so a small Gradio page is also mounted at /gradio.
Grading runs on CPU. Locally, just use: uvicorn app.main:app
"""

# `spaces` (ZeroGPU) must be imported before torch. It only exists on Hugging Face.
try:
    import spaces
except ImportError:
    spaces = None

import os  # noqa: E402
import threading  # noqa: E402

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402

from app.main import app  # noqa: E402
from grader.features import get_embedder  # noqa: E402

REPO_URL = "https://github.com/Aum07/ai-answer-sheet-grader"

if spaces is not None:

    @spaces.GPU
    def gpu_available() -> bool:
        """Required by ZeroGPU; grading doesn't use the GPU."""
        import torch

        return torch.cuda.is_available()


with gr.Blocks(title="AI Answer Sheet Grader") as status_page:
    gr.Markdown(
        f"# AI Answer Sheet Grader\nThe app lives at the [main page](/). Source: [GitHub]({REPO_URL})."
    )

app = gr.mount_gradio_app(app, status_page, path="/gradio")


def _warm_up() -> None:
    """Load the embedding model in the background so the first visitor doesn't wait for it."""
    try:
        get_embedder()
    except Exception as exc:
        print(f"Warm-up failed (will load on first request): {exc}")


if __name__ == "__main__":
    threading.Thread(target=_warm_up, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
