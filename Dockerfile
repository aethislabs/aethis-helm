# Aethis Helm API. The Linux torch wheel bundles CUDA, so one image runs on CPU (default) or GPU:
#   docker build -t aethis-helm .
#   docker run -p 8000:8000 --env-file .env aethis-helm                                  # CPU
#   docker run -p 8000:8000 --env-file .env -e HELM_DEVICE=cuda --gpus all aethis-helm   # GPU
# Mount a volume on /models to keep the downloaded backbone and weights between runs.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    HELM_HOST=0.0.0.0 \
    HELM_PORT=8000 \
    HELM_DEVICE=cpu

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . \
    && useradd --create-home --uid 1000 helm \
    && mkdir -p /models \
    && chown helm:helm /models

USER helm
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
    CMD python -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/health'))['model_loaded']"
CMD ["helm-serve"]
