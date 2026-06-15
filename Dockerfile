# VoiceForensics API container.
# NOTE: built/validated externally — this sandbox has no Docker daemon.
#   docker build -t voiceforensics .
#   docker run -p 8000:8000 voiceforensics
FROM python:3.12-slim AS base

# ffmpeg/ffprobe are required for decoding + metadata.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install CPU PyTorch first (smaller, no CUDA), then the package.
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install .

# Non-root runtime user.
RUN useradd -m appuser && mkdir -p /app/data/store /app/reports \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health').status_code==200 else 1)"

CMD ["uvicorn", "voiceforensics.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
