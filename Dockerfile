# ─────────────────────────────────────────────────────────────────────────────
#  Audio-RAG-Summarizer  –  Dockerfile
#
#  Base : nvidia/cuda 12.4 + cuDNN 9 + Ubuntu 22.04
#  Uses CUDA for Faster-Whisper (float16) and PyTorch inference.
#
#  Build:
#    docker build -t audio-rag .
#
#  Run:
#    docker run --gpus all --rm \
#      --env-file .env \
#      -v $(pwd)/data:/app/data \
#      -v $(pwd)/outputs:/app/outputs \
#      audio-rag "https://www.youtube.com/watch?v=XXXX"
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

# ── System dependencies ───────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        ffmpeg \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python / pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3      /usr/bin/pip

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only requirements first so Docker caches this layer
COPY requirements.txt .

# Install PyTorch with CUDA 12.4 wheels explicitly, then the rest
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project source ───────────────────────────────────────────────────────
COPY src/       ./src/
COPY main.py    .

# ── Create runtime directories ────────────────────────────────────────────────
RUN mkdir -p data/audio data/transcripts data/embeddings data/index \
             models \
             outputs/summaries outputs/logs

# ── Environment defaults (overridden at runtime via --env-file .env) ──────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_TOKEN="" \
    LOG_LEVEL="INFO"

# ── Entry point ───────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "main.py"]

# Default: show usage when no args are provided
CMD ["--help"]
