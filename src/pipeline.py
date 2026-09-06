"""
Core real-time pipeline: mic -> preprocessing -> STT -> confidence check
-> LLM -> TTS -> speakers.

OFFLINE GUARANTEE
------------------
Every network-shaped call in this file targets 127.0.0.1 only:
  - faster-whisper: loads model weights from the local HuggingFace cache
    / local model dir and runs inference on local GPU/CPU. Zero network
    calls once the model is downloaded (see verify_setup.py).
  - Ollama: talks to http://localhost:11434, a server this machine is
    running itself. No calls leave the box.
  - Piper: invokes a local `piper` binary against a local .onnx voice
    model file. No network involved at all.
To prove this to yourself: disable networking and re-run this file (see
README section 7) -- it should behave identically.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import time
import wave
from dataclasses import dataclass

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import webrtcvad

sys.path.insert(0, os.path.dirname(__file__))
from cuda_dlls import ensure_cuda_dlls_on_path  # noqa: E402

ensure_cuda_dlls_on_path()  # must run before faster_whisper/ctranslate2 is ever imported

from confidence_check import DEFAULT_CONFIDENCE_THRESHOLD, Decision, evaluate  # noqa: E402
from preprocessing import normalize_audio  # noqa: E402

# --------------------------------------------------------------------
# Config -- tune these for your hardware (see README section 5/6)
# --------------------------------------------------------------------
SAMPLE_RATE = 16000  # required by both webrtcvad and whisper
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples/frame
VAD_AGGRESSIVENESS = 2  # 0 (permissive) .. 3 (strict about what counts as speech)
SILENCE_FRAMES_TO_STOP = int(800 / FRAME_MS)   # ~0.8s trailing silence ends capture
MIN_SPEECH_FRAMES = int(150 / FRAME_MS)         # ignore blips shorter than this
MAX_RECORD_SECONDS = 15
PRE_SPEECH_PADDING_FRAMES = 10                   # ~300ms of audio kept before trigger

# faster-whisper: "small" fits comfortably on a 4GB GPU alongside a small
# quantized Ollama model. int8_float16 halves VRAM vs float16 with a
# negligible accuracy hit -- worth it when VRAM is the binding constraint.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
WHISPER_COMPUTE_TYPE_GPU = "int8_float16"
WHISPER_COMPUTE_TYPE_CPU = "int8"
BEAM_SIZE_FAST = 1     # first-pass greedy-ish decode, fast
BEAM_SIZE_RETRY = 5    # slower, wider search -- only used when confidence is low

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

def _default_piper_executable() -> str:
    """piper-tts installs a `piper`/`piper.exe` console script alongside
    the Python interpreter it was pip-installed into (venv/Scripts on
    Windows, venv/bin on Linux/Mac). That directory is only on PATH if
    the venv was `activate`d in the current shell -- resolving relative
    to sys.executable means it works either way, without relying on the
    caller having activated anything.
    """
    candidate = os.path.join(
        os.path.dirname(sys.executable), "piper.exe" if os.name == "nt" else "piper"
    )
    return candidate if os.path.exists(candidate) else "piper"


PIPER_EXECUTABLE = os.environ.get("PIPER_EXECUTABLE", _default_piper_executable())
PIPER_MODEL_PATH = os.environ.get(
    "PIPER_MODEL_PATH", os.path.join("models", "piper", "en_US-lessac-medium.onnx")
)

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")

logger = logging.getLogger("voice_chatbot.pipeline")


def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


@dataclass
class StageTimings:
    capture_s: float = 0.0
    preprocess_s: float = 0.0
    stt_s: float = 0.0
    llm_s: float = 0.0
    tts_s: float = 0.0

    def log(self) -> None:
        logger.info(
            "Timings -- capture=%.2fs preprocess=%.3fs stt=%.2fs llm=%.2fs tts=%.2fs total=%.2fs",
            self.capture_s,
            self.preprocess_s,
            self.stt_s,
            self.llm_s,
            self.tts_s,
            self.capture_s + self.preprocess_s + self.stt_s + self.llm_s + self.tts_s,
        )


# --------------------------------------------------------------------
# 1. Audio capture with voice-activity detection
# --------------------------------------------------------------------
def capture_audio(max_seconds: float = MAX_RECORD_SECONDS) -> np.ndarray | None:
    """Record from the default mic until the speaker stops talking.

    Uses WebRTC's VAD (frame-by-frame speech/silence classification) so
    we don't record a fixed duration -- recording starts once real speech
    is detected and ends after a trailing-silence window, which keeps
    latency down for short commands and avoids truncating longer ones.

    Returns float32 mono audio at SAMPLE_RATE, or None if nothing was said.
    """
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_q: queue.Queue[bytes] = queue.Queue()

    def _callback(indata, frames, time_info, status):
        if status:
            logger.warning("Audio input status: %s", status)
        frame_q.put(bytes(indata))

    ring_buffer: list[bytes] = []
    voiced_frames: list[bytes] = []
    triggered = False
    silent_run = 0
    speech_frame_count = 0
    start_time = time.time()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
        channels=1,
        callback=_callback,
    ):
        logger.info("Listening...")
        while True:
            if time.time() - start_time > max_seconds:
                logger.info("Max recording duration hit, stopping capture.")
                break
            try:
                frame = frame_q.get(timeout=1.0)
            except queue.Empty:
                continue

            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if not triggered:
                ring_buffer.append(frame)
                if len(ring_buffer) > PRE_SPEECH_PADDING_FRAMES:
                    ring_buffer.pop(0)
                if is_speech:
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
                    speech_frame_count = 1
            else:
                voiced_frames.append(frame)
                if is_speech:
                    silent_run = 0
                    speech_frame_count += 1
                else:
                    silent_run += 1
                    if silent_run >= SILENCE_FRAMES_TO_STOP:
                        break

    if not triggered or speech_frame_count < MIN_SPEECH_FRAMES:
        logger.info("No sustained speech detected.")
        return None

    raw = b"".join(voiced_frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


# --------------------------------------------------------------------
# 2. STT (faster-whisper, local, GPU-accelerated with CPU fallback)
# --------------------------------------------------------------------
def _smoke_test_transcribe(model) -> None:
    """Run one cheap transcription to force the encoder to actually
    execute on whatever device it was constructed with. Raises if the
    device path is broken (e.g. a missing CUDA DLL), so the caller can
    catch it and fall back before the user's first real utterance does.
    """
    silence = np.zeros(SAMPLE_RATE // 4, dtype=np.float32)  # 0.25s
    segments, _info = model.transcribe(silence, language="en")
    list(segments)  # force the generator to actually run inference


def load_whisper_model():
    """Load faster-whisper once at startup. Tries GPU first (fast path
    for this project's whole reason for existing); falls back to CPU so
    the chatbot still works on a machine without a usable CUDA setup.

    Note: WhisperModel(device="cuda") can construct successfully even
    when the GPU path is actually broken (e.g. missing cuBLAS/cuDNN DLLs
    on Windows) -- the failure only surfaces on the first real inference
    call. So the GPU path is verified with a cheap smoke-test transcribe
    here rather than trusting construction alone; on this project's own
    4GB-VRAM dev machine that smoke test is what catches it and routes
    to CPU instead of crashing on the first real utterance.
    """
    from faster_whisper import WhisperModel  # local import: heavy, load once

    try:
        model = WhisperModel(
            WHISPER_MODEL_SIZE, device="cuda", compute_type=WHISPER_COMPUTE_TYPE_GPU
        )
        _smoke_test_transcribe(model)
        logger.info(
            "faster-whisper loaded on GPU (model=%s, compute_type=%s)",
            WHISPER_MODEL_SIZE,
            WHISPER_COMPUTE_TYPE_GPU,
        )
        return model, "cuda"
    except Exception as exc:
        logger.warning("GPU path unusable (%s), falling back to CPU.", exc)
        model = WhisperModel(
            WHISPER_MODEL_SIZE, device="cpu", compute_type=WHISPER_COMPUTE_TYPE_CPU
        )
        logger.info("faster-whisper loaded on CPU (model=%s)", WHISPER_MODEL_SIZE)
        return model, "cpu"


def transcribe(model, audio: np.ndarray, beam_size: int = BEAM_SIZE_FAST):
    """Runs faster-whisper on already-preprocessed audio. Returns
    (text, segments_list) -- segments carry the avg_logprob confidence
    scores the confidence_check layer needs.
    """
    segments_iter, _info = model.transcribe(
        audio,
        language="en",
        beam_size=beam_size,
        vad_filter=False,  # we already did VAD on capture; avoid double-gating
    )
    segments = list(segments_iter)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, segments


def transcribe_with_confidence(model, audio: np.ndarray):
    """Implements the confidence-aware retry policy from confidence_check.py:
    fast greedy decode -> if low-confidence, retry once with a wider beam
    search -> if still low-confidence, give up and signal ASK_TO_REPEAT
    rather than passing a guess to the LLM.
    """
    text, segments = transcribe(model, audio, beam_size=BEAM_SIZE_FAST)
    result = evaluate(
        segments, text, beam_size_used=BEAM_SIZE_FAST, attempt=1, is_retry_attempt=False
    )
    if result.decision == Decision.ACCEPTED:
        return text, result

    logger.info("Low confidence on first pass, retrying with beam_size=%d", BEAM_SIZE_RETRY)
    text, segments = transcribe(model, audio, beam_size=BEAM_SIZE_RETRY)
    result = evaluate(
        segments, text, beam_size_used=BEAM_SIZE_RETRY, attempt=2, is_retry_attempt=True
    )
    return text, result


# --------------------------------------------------------------------
# 3. LLM (Ollama, local server on localhost:11434)
# --------------------------------------------------------------------
# Without guidance, instruction-tuned models default to chat-app-style
# answers: numbered lists, headers, multi-paragraph explanations. That's
# fine on a screen and actively bad here -- every response gets read
# aloud by Piper, so a 4-point numbered list becomes a wall of spoken
# text with no visual structure to lean on. This system prompt is what
# actually fixes that (found while comparing models: swapping models
# didn't fix verbosity, this did -- see README known limitations).
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your replies are converted to "
    "speech and spoken aloud, so keep answers short: 1-3 plain "
    "conversational sentences. Never use lists, headers, markdown, or "
    "any formatting that only makes sense in writing."
)


def generate_response(prompt: str, model: str = OLLAMA_MODEL, on_token=None) -> str:
    """Streams a response from a local Ollama server. `on_token`, if
    given, is called with each incremental chunk (used by the CLI to
    print/speak as it arrives instead of waiting for the full reply).
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "system": SYSTEM_PROMPT, "stream": True}
    full_text = []
    try:
        # 180s headroom covers Ollama's one-time cold-load of model weights
        # into memory on the first call; warm calls return in a couple seconds.
        with requests.post(url, json=payload, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    full_text.append(token)
                    if on_token:
                        on_token(token)
                if chunk.get("done"):
                    break
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not reach Ollama at "
            f"{OLLAMA_URL}. Is `ollama serve` running? ({exc})"
        ) from exc
    return "".join(full_text).strip()


# --------------------------------------------------------------------
# 4. TTS (Piper, local binary + local voice model)
# --------------------------------------------------------------------
def synthesize_speech(text: str, out_path: str = os.path.join(LOG_DIR, "_tts_out.wav")) -> str:
    """Runs the local `piper` executable against a local voice model
    file. Piper reads text on stdin and writes a wav file -- no network
    call of any kind is involved.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = [PIPER_EXECUTABLE, "--model", PIPER_MODEL_PATH, "--output_file", out_path]
    proc = subprocess.run(
        cmd, input=text.encode("utf-8"), capture_output=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Piper synthesis failed (exit {proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')}"
        )
    return out_path


def play_audio(path: str) -> None:
    data, sr = sf.read(path, dtype="float32")
    sd.play(data, sr)
    sd.wait()


# --------------------------------------------------------------------
# Main conversation loop
# --------------------------------------------------------------------
EXIT_WORDS = {"exit", "quit", "goodbye", "stop chatbot"}


def run():
    setup_logging()
    logger.info("Starting voice chatbot (faster-whisper + Ollama + Piper, all local).")
    whisper_model, device = load_whisper_model()
    print(f"[pipeline] Whisper running on: {device}")
    print("[pipeline] Speak now. Say 'exit' to quit. Ctrl+C also works.")

    try:
        while True:
            timings = StageTimings()

            t0 = time.time()
            audio = capture_audio()
            timings.capture_s = time.time() - t0
            if audio is None:
                continue

            t0 = time.time()
            try:
                processed_audio, report = normalize_audio(audio, SAMPLE_RATE)
                logger.info("Preprocessing: %s", report)
            except Exception as exc:
                # Belt-and-suspenders: preprocessing.py already guards its
                # own steps individually, but a live conversation loop
                # should never die because an enhancement step misbehaved.
                # Fall back to the raw, unprocessed audio for this turn.
                logger.warning("Preprocessing raised unexpectedly (%s); using raw audio.", exc)
                processed_audio = audio
            timings.preprocess_s = time.time() - t0

            t0 = time.time()
            text, conf_result = transcribe_with_confidence(whisper_model, processed_audio)
            timings.stt_s = time.time() - t0

            if conf_result.decision == Decision.ASK_TO_REPEAT:
                print("[pipeline] Low confidence -- asking user to repeat.")
                t0 = time.time()
                wav = synthesize_speech("Sorry, could you say that again?")
                play_audio(wav)
                timings.tts_s = time.time() - t0
                timings.log()
                continue

            print(f"You said: {text}  (confidence={conf_result.avg_logprob:.2f})")
            if text.strip().lower() in EXIT_WORDS:
                print("[pipeline] Exit word heard, shutting down.")
                break
            if not text.strip():
                continue

            t0 = time.time()
            response = generate_response(text, on_token=lambda tok: print(tok, end="", flush=True))
            print()
            timings.llm_s = time.time() - t0

            t0 = time.time()
            wav = synthesize_speech(response)
            play_audio(wav)
            timings.tts_s = time.time() - t0

            timings.log()
    except KeyboardInterrupt:
        print("\n[pipeline] Interrupted, shutting down.")


if __name__ == "__main__":
    run()
