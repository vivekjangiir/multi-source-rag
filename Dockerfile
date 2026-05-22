# ─────────────────────────────────────────────
#  Multi-Source RAG — Hugging Face Spaces
#  Docker image (Python 3.11 slim)
# ─────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# System deps needed by ChromaDB / sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ca-certificates \
    openssl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies first (layer-cached) ──────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade certifi requests

# ── Pre-download the embedding model at BUILD time ─────────────
# This caches the ~80 MB model into the image so ingestion never
# times out waiting for a download at runtime.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Embedding model cached OK')"

# ── Copy application code ──────────────────────────────────────
COPY . .

# ── Persistent storage ─────────────────────────────────────────
# HF Spaces mounts /data as a persistent volume (Docker spaces).
# We point ChromaDB there so vectors survive restarts.
RUN mkdir -p /data/chroma_db

# ── Runtime config ─────────────────────────────────────────────
ENV PORT=7860
ENV CHROMA_PERSIST_DIR=/data/chroma_db
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
# Point requests/urllib to the system cert bundle so YouTube SSL works
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
