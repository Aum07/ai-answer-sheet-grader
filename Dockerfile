FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/.cache/nltk \
    PORT=7860

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build
COPY requirements.txt .
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Bake model weights into the image so the first request is not a download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')" \
    && python -c "import nltk; [nltk.download(r, quiet=True) for r in ('punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng')]"

COPY grader ./grader
COPY app ./app
COPY models ./models
COPY samples ./samples

RUN useradd -m appuser && chown -R appuser /app
USER appuser

# 7860 is the Hugging Face Spaces default; override with -e PORT=8000 elsewhere
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
