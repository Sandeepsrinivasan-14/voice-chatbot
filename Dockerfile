# syntax=docker/dockerfile:1
#
# CPU image for the browser chat UI (src/chat_web.py).
#
#   docker build -t voice-chatbot .
#   docker run --rm -p 127.0.0.1:5006:5006 \
#     -e OLLAMA_URL=http://host.docker.internal:11434 \
#     -v "$PWD/models:/app/models" -v whisper-cache:/home/app/.cache \
#     voice-chatbot
#
# The container needs no microphone or speakers: audio is recorded and played
# back in your browser. Ollama is NOT bundled -- point OLLAMA_URL at one you run
# (on Linux add `--add-host=host.docker.internal:host-gateway`). Piper voices
# are not baked in; mount them at /app/models/piper (see models/piper/README.md).
# GPU inference is out of scope for this image (use the host install for CUDA).
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libportaudio2: sounddevice loads it at import time even though no audio
# device is used. libsndfile1: soundfile backend.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libportaudio2 libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY --chown=app:app src ./src
RUN mkdir -p logs data/learning models/piper && chown -R app:app /app

USER app

# 0.0.0.0 is required inside a container; the app has NO authentication, so
# always publish the port to loopback only (-p 127.0.0.1:5006:5006).
ENV WEB_HOST=0.0.0.0 \
    CHAT_WEB_PORT=5006 \
    PRODUCTION=1

EXPOSE 5006

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${CHAT_WEB_PORT}/api/health" || exit 1

CMD ["python", "src/chat_web.py"]
