import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: similar wording -> similar vectors.

    Lets the feature pipeline and model run in tests without downloading mpnet.
    """

    dim = 256

    def encode(self, sentences, **kwargs):
        out = np.zeros((len(sentences), self.dim))
        for i, s in enumerate(sentences):
            for word in s.lower().split():
                h = int(hashlib.md5(word.strip(".,!?").encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return out


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


def make_pdf(path: Path, lines: list[str]) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in lines:
        c.drawString(50, y, line)
        y -= 16
    c.save()
    return path


@pytest.fixture
def pdf_factory(tmp_path):
    def _make(name: str, lines: list[str]) -> Path:
        return make_pdf(tmp_path / name, lines)

    return _make
