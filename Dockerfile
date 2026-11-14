FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/models PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY manga_translator manga_translator
COPY assets assets

RUN useradd --create-home --uid 1000 app && mkdir -p /library /models && chown app /library /models
USER app

EXPOSE 8000
VOLUME ["/library", "/models"]
CMD ["python", "-m", "manga_translator", "serve", "--host", "0.0.0.0", "--port", "8000", "--library", "/library"]
