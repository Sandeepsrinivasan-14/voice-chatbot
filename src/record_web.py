"""
Phase 4 helper (browser variant): a local, good-looking recorder for the
modulation-robustness test dataset -- an alternative front end to
record_test_samples.py's terminal prompts, same output.

Runs a Flask server bound to 127.0.0.1 only. The browser tab is just a
UI shell: your mic audio never leaves this machine. The page records
with MediaRecorder (webm/opus, whatever the browser gives us), POSTs
the blob to /api/upload, and this server decodes + resamples it to
16kHz mono float32 with PyAV (already a faster-whisper dependency --
no new decoding library needed) before writing the exact same
data/test_samples/{word}_{modulation}.wav + manifest.csv format that
benchmark.py already expects. The two recorders are interchangeable;
use whichever you like better.

Usage:
    python src/record_web.py
    -> open http://localhost:5005 in a browser
"""

from __future__ import annotations

import csv
import os
import sys

import soundfile as sf
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from browser_audio import decode_browser_audio  # noqa: E402
from config import CONFIG  # noqa: E402
from web_common import register_error_handlers, run_app  # noqa: E402

SAMPLE_RATE = CONFIG.sample_rate
DEFAULT_WORDS = ["yes", "no", "stop", "help", "start"]
MODULATIONS = ["normal", "whispered", "shouted", "slow", "fast"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "test_samples")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.csv")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_recorder")
MIN_CLIP_SECONDS = 0.3

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = CONFIG.max_upload_bytes
register_error_handlers(app)


def load_manifest_rows() -> list[dict]:
    if not os.path.exists(MANIFEST_PATH):
        return []
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_manifest_rows(rows: list[dict]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "expected_word", "modulation"])
        writer.writeheader()
        writer.writerows(rows)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/plan")
def plan():
    """The full (word, modulation) grid plus which combos already have a
    saved recording, so reloading the page resumes instead of restarting.
    """
    words_param = request.args.get("words")
    word_list = [w.strip() for w in words_param.split(",") if w.strip()] if words_param else DEFAULT_WORDS
    done = {(r["expected_word"], r["modulation"]) for r in load_manifest_rows()}
    combos = [
        {"word": w, "modulation": m, "done": [w, m] in [list(d) for d in done]}
        for w in word_list
        for m in MODULATIONS
    ]
    # the list-comparison above is O(n^2) but n is tiny (<=25); correctness > cleverness here
    return jsonify({"combos": combos, "words": word_list, "modulations": MODULATIONS})


@app.route("/api/upload", methods=["POST"])
def upload():
    word = request.form.get("word", "").strip()
    modulation = request.form.get("modulation", "").strip()
    if not word or modulation not in MODULATIONS:
        return jsonify({"error": "Missing or invalid word/modulation."}), 400
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request."}), 400

    raw_bytes = request.files["audio"].read()
    try:
        audio = decode_browser_audio(raw_bytes, SAMPLE_RATE)
    except Exception as exc:
        return jsonify({"error": f"Could not decode audio: {exc}"}), 400

    if audio.size < int(SAMPLE_RATE * MIN_CLIP_SECONDS):
        return jsonify({"error": "Recording too short -- try again."}), 400

    filename = f"{word}_{modulation}.wav"
    filepath = os.path.join(OUT_DIR, filename)
    os.makedirs(OUT_DIR, exist_ok=True)
    sf.write(filepath, audio, SAMPLE_RATE)

    rows = [
        r for r in load_manifest_rows()
        if not (r["expected_word"] == word and r["modulation"] == modulation)
    ]
    rows.append({"filename": filename, "expected_word": word, "modulation": modulation})
    save_manifest_rows(rows)

    return jsonify({"ok": True, "filename": filename, "duration_s": round(audio.size / SAMPLE_RATE, 2)})


@app.route("/api/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Saving into: {OUT_DIR}")
    print("Everything here stays on this machine -- close the tab/Ctrl+C to stop.")
    run_app(app, host="127.0.0.1", port=CONFIG.record_web_port, name="record_web")
