"""
A browser-based voice chat front end for the pipeline in pipeline.py --
this is the "actually talk to the chatbot" experience, as opposed to
src/record_web.py (which only builds the benchmark test set) or
src/cli.py (the terminal version of the same conversation loop).

Runs a Flask server bound to 127.0.0.1 only. It reuses pipeline.py's
exact functions (load_whisper_model, transcribe_with_confidence,
synthesize_speech, SYSTEM_PROMPT) so the web UI behaves identically to
the CLI -- same confidence gating, same GPU/CPU fallback, same
confidence_log.csv auditing -- it's a different front end on the same
back end, not a separate reimplementation.

Everything here is local:
  - faster-whisper: local model, local GPU/CPU (via pipeline.py)
  - Ollama: http://localhost:11434, a server this machine runs itself
  - Piper: local binary + local voice model (via pipeline.py)
The browser tab is just a UI shell -- audio and text never leave this
machine.

Usage:
    python src/chat_web.py
    -> open http://localhost:5006 in a browser
"""

from __future__ import annotations

import json
import os
import sys
import time

import requests
from flask import Flask, Response, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from browser_audio import decode_browser_audio  # noqa: E402
from config import CONFIG  # noqa: E402
import pipeline  # noqa: E402  (also wires up cuda_dlls before faster_whisper loads)
from confidence_check import Decision  # noqa: E402
from preprocessing import normalize_audio  # noqa: E402
from web_common import register_error_handlers, run_app  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_chat")
MAX_HISTORY_MESSAGES = 12  # ~6 exchanges of context sent back to Ollama each turn

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = CONFIG.max_upload_bytes
register_error_handlers(app)
_whisper_model = None
_whisper_device = None


def get_whisper_model():
    """Loaded once, lazily, on first request rather than at import time --
    keeps `python src/chat_web.py --help`-style invocations fast and
    means a broken model load surfaces as a clear HTTP error, not a
    server that never starts.
    """
    global _whisper_model, _whisper_device
    if _whisper_model is None:
        _whisper_model, _whisper_device = pipeline.load_whisper_model()
        print(f"[chat_web] faster-whisper ready on {_whisper_device}")
    return _whisper_model


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "whisper_device": _whisper_device})


@app.route("/api/transcribe", methods=["POST"])
def transcribe_endpoint():
    """Audio in, transcription + confidence decision out. Mirrors
    pipeline.run()'s capture -> preprocess -> STT -> confidence-check
    stages exactly (same function calls), just fed a browser recording
    instead of a live mic stream.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request."}), 400

    raw_bytes = request.files["audio"].read()
    try:
        audio = decode_browser_audio(raw_bytes, pipeline.SAMPLE_RATE)
    except Exception as exc:
        return jsonify({"error": f"Could not decode audio: {exc}"}), 400

    if audio.size < int(pipeline.SAMPLE_RATE * 0.2):
        return jsonify({"error": "Recording too short -- try again."}), 400

    try:
        processed_audio, _report = normalize_audio(audio, pipeline.SAMPLE_RATE)
    except Exception as exc:
        # Preprocessing is an enhancement, never a hard requirement -- same
        # belt-and-suspenders fallback as pipeline.py's run() loop.
        app.logger.warning("Preprocessing raised unexpectedly (%s); using raw audio.", exc)
        processed_audio = audio

    model = get_whisper_model()
    text, conf_result = pipeline.transcribe_with_confidence(model, processed_audio)

    return jsonify({
        "text": text,
        "decision": conf_result.decision.value,
        "avg_logprob": conf_result.avg_logprob,
        "threshold": conf_result.threshold,
        "accepted": conf_result.decision == Decision.ACCEPTED,
        "whisper_device": _whisper_device,
    })


@app.route("/api/respond", methods=["POST"])
def respond_endpoint():
    """Streams the LLM's reply back as plain text chunks, proxying
    Ollama's own streaming /api/chat response. Uses the chat endpoint
    (message history) rather than pipeline.generate_response()'s
    single-turn /api/generate, so the web UI gets real multi-turn memory
    -- an upgrade over the CLI's stateless-per-turn design, made
    possible because the browser can hold the running history for us.
    """
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    history = data.get("history") or []
    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    messages = [{"role": "system", "content": pipeline.SYSTEM_PROMPT}]
    messages.extend(history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": user_message})

    def generate():
        # Retries a connection failure that happens before any token was
        # yielded (a transient hiccup) -- same policy as, and for the same
        # reason as, pipeline.generate_response()'s retry logic: once a
        # token has already reached the client it's too late to restart
        # without duplicating output.
        last_exc: Exception | None = None
        for attempt in range(1, CONFIG.ollama_max_retries + 2):
            yielded_any = False
            try:
                with requests.post(
                    f"{pipeline.OLLAMA_URL}/api/chat",
                    json={"model": pipeline.OLLAMA_MODEL, "messages": messages, "stream": True},
                    stream=True,
                    timeout=CONFIG.ollama_timeout_s,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yielded_any = True
                            yield token
                        if chunk.get("done"):
                            break
                return
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                if yielded_any or attempt > CONFIG.ollama_max_retries:
                    break
                time.sleep(0.5 * attempt)
            except Exception as exc:  # noqa: BLE001 -- surface any failure as visible chat text
                yield f"\n[Error talking to Ollama: {exc}]"
                return

        yield (
            f"\n[Could not reach Ollama at {pipeline.OLLAMA_URL} -- "
            f"is `ollama serve` running? ({last_exc})]"
        )

    return Response(generate(), mimetype="text/plain")


@app.route("/api/tts", methods=["POST"])
def tts_endpoint():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Empty text."}), 400
    try:
        wav_path = pipeline.synthesize_speech(text)
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        return Response(wav_bytes, mimetype="audio/wav")
    except Exception as exc:
        return jsonify({"error": f"TTS failed: {exc}"}), 500


if __name__ == "__main__":
    print("Loading faster-whisper on first request (not at startup) -- the")
    print("first message you send will take a moment longer than the rest.")
    print("Everything here stays on this machine -- close the tab/Ctrl+C to stop.")
    run_app(app, host="127.0.0.1", port=CONFIG.chat_web_port, name="chat_web")
