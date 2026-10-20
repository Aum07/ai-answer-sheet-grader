"""Hugging Face Spaces entry point.

Free ZeroGPU hardware only accepts a Space whose web server is started by Gradio's launch()
and that declares a @spaces.GPU function. So Gradio starts the server, and the project's own
website and API (app/main.py) are then placed in front of Gradio's routes: visitors get the
normal site at "/". Grading runs on CPU. Locally, just use: uvicorn app.main:app
"""

import os

# ZeroGPU pretends a GPU exists; the grader is CPU-only, so pin the embedding model to the CPU.
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")

# `spaces` (ZeroGPU) must be imported before torch. It only exists on Hugging Face.
try:
    import spaces
except ImportError:
    spaces = None

import threading  # noqa: E402

import gradio as gr  # noqa: E402

from app.main import app as site  # noqa: E402
from grader import scoring  # noqa: E402
from grader.features import get_embedder  # noqa: E402

REPO_URL = "https://github.com/Aum07/ai-answer-sheet-grader"

if spaces is not None:

    @spaces.GPU
    def gpu_available() -> bool:
        """Required by ZeroGPU; grading doesn't use the GPU."""
        import torch

        return torch.cuda.is_available()


with gr.Blocks(title="AI Answer Sheet Grader") as placeholder:
    gr.Markdown(f"# AI Answer Sheet Grader\nSee the [main page](/) or the [source on GitHub]({REPO_URL}).")


def _warm_up() -> None:
    """Load the embedding model in the background so the first visitor doesn't wait for it."""
    try:
        get_embedder()
    except Exception as exc:
        print(f"Warm-up failed (will load on first request): {exc}")


if __name__ == "__main__":
    placeholder.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False, prevent_thread_lock=True)

    # Our routes go first, so "/" and the rest serve the project's own site; Gradio's internal
    # /gradio_api routes stay available behind them.
    placeholder.app.router.routes[0:0] = list(site.router.routes)
    scoring.load_model()  # normally done by the site's own startup hook
    threading.Thread(target=_warm_up, daemon=True).start()
    print("Site routes mounted in front of Gradio")

    placeholder.block_thread()
