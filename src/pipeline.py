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
  - Piper: runs in-process via `piper.voice.PiperVoice` against a local
    .onnx voice model file. No network involved at all.
To prove this to yourself: disable networking and re-run this file (see
README section 7) -- it should behave identically.
"""

from __future__ import annotations

import atexit
import json
import logging
import logging.handlers
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from piper.voice import PiperVoice

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import webrtcvad

sys.path.insert(0, os.path.dirname(__file__))
from cuda_dlls import ensure_cuda_dlls_on_path  # noqa: E402

ensure_cuda_dlls_on_path()  # must run before faster_whisper/ctranslate2 is ever imported

from config import CONFIG  # noqa: E402
from confidence_check import Decision, evaluate  # noqa: E402
from preprocessing import normalize_audio  # noqa: E402

# --------------------------------------------------------------------
# Config -- these are aliases onto the single source of truth in
# config.py (kept as module attributes here since cli.py, chat_web.py,
# and benchmark.py already reference e.g. `pipeline.OLLAMA_URL` --
# changing *where the value is read from* shouldn't require touching
# every call site). To actually change a value, edit config.py / your
# .env, not these lines.
# --------------------------------------------------------------------
SAMPLE_RATE = CONFIG.sample_rate  # required by both webrtcvad and whisper
FRAME_MS = CONFIG.frame_ms
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples/frame
VAD_AGGRESSIVENESS = CONFIG.vad_aggressiveness  # 0 (permissive) .. 3 (strict)
SILENCE_FRAMES_TO_STOP = int(800 / FRAME_MS)  # ~0.8s trailing silence ends capture
MIN_SPEECH_FRAMES = int(150 / FRAME_MS)  # ignore blips shorter than this
MAX_RECORD_SECONDS = CONFIG.max_record_seconds
PRE_SPEECH_PADDING_FRAMES = 10  # ~300ms of audio kept before trigger

WHISPER_MODEL_SIZE = CONFIG.whisper_model_size
WHISPER_COMPUTE_TYPE_GPU = CONFIG.whisper_compute_type_gpu
WHISPER_COMPUTE_TYPE_CPU = CONFIG.whisper_compute_type_cpu
BEAM_SIZE_FAST = CONFIG.beam_size_fast
BEAM_SIZE_RETRY = CONFIG.beam_size_retry
WHISPER_LANGUAGE = CONFIG.whisper_language  # "auto" or a forced "en"/"hi"/"ta"

OLLAMA_URL = CONFIG.ollama_url
OLLAMA_MODEL = CONFIG.ollama_model

PIPER_EXECUTABLE = CONFIG.piper_executable  # kept only for verify_setup.py's own CLI smoke test;
# synthesize_speech() below no longer shells out to it (see _load_piper_voice).
PIPER_MODEL_PATH = CONFIG.piper_model_path
PIPER_VOICES = CONFIG.piper_voices  # {"en": ..., "hi": ...} -- Piper-speakable languages

LOG_DIR = CONFIG.log_dir
LOG_FILE = CONFIG.pipeline_log_path

logger = logging.getLogger("voice_chatbot.pipeline")


def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    # RotatingFileHandler instead of a plain FileHandler: a chatbot left
    # running for a while (or a busy chat_web.py server) would otherwise
    # grow pipeline.log without bound. Caps it at a few rotated files.
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=CONFIG.log_max_bytes,
        backupCount=CONFIG.log_backup_count,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[file_handler, logging.StreamHandler(sys.stdout)],
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


def load_whisper_model_from_path(model_path: str):
    """Same GPU-first/CPU-fallback loading as load_whisper_model(), but for
    an arbitrary local CTranslate2 model directory instead of
    CONFIG.whisper_model_size -- used to load a continuous-learning
    candidate/production model (see learning/model_registry.py) rather
    than a stock faster-whisper size name. faster-whisper's WhisperModel
    accepts either interchangeably, so this only differs from
    load_whisper_model() in not reading WHISPER_MODEL_SIZE/compute-type
    config at all: the caller passes the exact path it wants loaded.
    """
    from faster_whisper import WhisperModel  # local import: heavy, load once

    try:
        model = WhisperModel(model_path, device="cuda", compute_type=WHISPER_COMPUTE_TYPE_GPU)
        _smoke_test_transcribe(model)
        logger.info("faster-whisper loaded on GPU from %s", model_path)
        return model, "cuda"
    except Exception as exc:
        logger.warning("GPU path unusable for %s (%s), falling back to CPU.", model_path, exc)
        model = WhisperModel(model_path, device="cpu", compute_type=WHISPER_COMPUTE_TYPE_CPU)
        logger.info("faster-whisper loaded on CPU from %s", model_path)
        return model, "cpu"


def transcribe(
    model, audio: np.ndarray, beam_size: int = BEAM_SIZE_FAST, language: str | None = None
):
    """Runs faster-whisper on already-preprocessed audio. Returns
    (text, segments_list, detected_language) -- segments carry the
    avg_logprob confidence scores the confidence_check layer needs.

    `language` is a forced ISO code ("en"/"hi"/"ta"/...), or None/"auto" to
    let faster-whisper detect the spoken language itself -- this is what
    makes Hindi/Tamil (not just English) transcribable; forcing a language
    used to be hardcoded to "en" here.
    """
    forced_language = None if language in (None, "auto") else language
    segments_iter, info = model.transcribe(
        audio,
        language=forced_language,
        beam_size=beam_size,
        vad_filter=False,  # we already did VAD on capture; avoid double-gating
    )
    segments = list(segments_iter)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, segments, info.language


def transcribe_with_confidence(model, audio: np.ndarray, language: str | None = None):
    """Implements the confidence-aware retry policy from confidence_check.py:
    fast greedy decode -> if low-confidence, retry once with a wider beam
    search -> if still low-confidence, give up and signal ASK_TO_REPEAT
    rather than passing a guess to the LLM.

    `language` forces a language ("en"/"hi"/"ta") instead of relying on
    detection; defaults to CONFIG.whisper_language ("auto" out of the box).
    Returns (text, result, detected_language) -- detected_language is
    whatever faster-whisper actually identified (or the forced language).
    """
    requested_language = WHISPER_LANGUAGE if language is None else language
    text, segments, detected_language = transcribe(
        model, audio, beam_size=BEAM_SIZE_FAST, language=requested_language
    )
    result = evaluate(
        segments, text, beam_size_used=BEAM_SIZE_FAST, attempt=1, is_retry_attempt=False
    )
    if result.decision == Decision.ACCEPTED:
        return text, result, detected_language

    logger.info("Low confidence on first pass, retrying with beam_size=%d", BEAM_SIZE_RETRY)
    # Force whatever language the first pass settled on, so the retry can't
    # flip-flop to a different language guess on the same audio.
    retry_language = detected_language if requested_language == "auto" else requested_language
    text, segments, detected_language = transcribe(
        model, audio, beam_size=BEAM_SIZE_RETRY, language=retry_language
    )
    result = evaluate(
        segments, text, beam_size_used=BEAM_SIZE_RETRY, attempt=2, is_retry_attempt=True
    )
    return text, result, detected_language


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
    "any formatting that only makes sense in writing. Always reply in "
    "the same language the user just spoke or wrote in -- English, "
    "Hindi, or Tamil -- never translate to a different language than "
    "the one you were addressed in."
)


def generate_response(prompt: str, model: str = OLLAMA_MODEL, on_token=None) -> str:
    """Streams a response from a local Ollama server. `on_token`, if
    given, is called with each incremental chunk (used by the CLI to
    print/speak as it arrives instead of waiting for the full reply).

    Retries a couple of times on a connection failure that happens before
    any token was streamed (a transient hiccup -- Ollama mid-model-swap,
    a brief cold-start race) -- but never retries once part of the reply
    has already reached `on_token`, since replaying from scratch at that
    point would duplicate output the caller has already acted on.
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "system": SYSTEM_PROMPT, "stream": True}
    last_exc: Exception | None = None

    for attempt in range(1, CONFIG.ollama_max_retries + 2):  # +2: first try + N retries
        full_text: list[str] = []
        try:
            # ollama_timeout_s headroom covers Ollama's one-time cold-load of
            # model weights into memory on the first call; warm calls return
            # in a couple seconds.
            with requests.post(
                url, json=payload, stream=True, timeout=CONFIG.ollama_timeout_s
            ) as resp:
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
            return "".join(full_text).strip()
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            if full_text or attempt > CONFIG.ollama_max_retries:
                break
            logger.warning(
                "Ollama connection failed (attempt %d/%d), retrying...",
                attempt,
                CONFIG.ollama_max_retries + 1,
            )
            time.sleep(0.5 * attempt)

    raise RuntimeError(
        f"Could not reach Ollama at {OLLAMA_URL}. Is `ollama serve` running? ({last_exc})"
    ) from last_exc


# --------------------------------------------------------------------
# 4. TTS (Piper, in-process voice model)
# --------------------------------------------------------------------
# SYSTEM_PROMPT above only *asks* the LLM for "1-3 plain conversational
# sentences" -- it has no way to enforce that, and on a confusing/garbled
# input the model can ignore it and ramble for several paragraphs. That's
# merely awkward for Piper/indic_tts.py (both fast, non-autoregressive
# models now -- see indic_tts.py's docstring for why Tamil switched off
# indic-parler-tts), but still worth a hard cap: an unbounded reply is an
# unbounded wait no matter how fast the model is per character. So this
# is enforced here, not just requested in the prompt: speech is capped at
# MAX_SPEECH_CHARS, cut at the last sentence boundary within that limit
# (never mid-sentence) so what IS spoken still sounds complete.
MAX_SPEECH_CHARS = 280
# Tamil gets a tighter cap too, but its job is a per-CHUNK safety net, not
# a whole-reply truncation: chat_web.py's index.html sentence-chunks a
# reply CLIENT-SIDE before ever calling /api/tts, so in the common case
# this function already receives one sentence at a time. This only
# matters for one pathological run-on "sentence" with no punctuation
# before the next boundary. 180 comfortably fits one real full sentence
# (90 often cut a real sentence off mid-thought). Like SPEECH_MARGIN /
# VAD_RMS_ABS_FLOOR in index.html, this was tuned by hand, not derived --
# adjust it if your own hardware differs.
MAX_SPEECH_CHARS_TA = 180
_SENTENCE_END_RE = re.compile(r"[.!?।॥]")  # latin + Devanagari/Indic full stops


def truncate_for_speech(text: str, max_chars: int = MAX_SPEECH_CHARS) -> str:
    """Cuts `text` down to at most `max_chars`, preferring to end on the
    last sentence boundary found within that window so speech never
    trails off mid-sentence. Falls back to a hard cut if no sentence
    boundary exists in range (e.g. one long run-on sentence).
    """
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    boundaries = [m.end() for m in _SENTENCE_END_RE.finditer(window)]
    if boundaries:
        return window[: boundaries[-1]].strip()
    return window.rstrip() + "..."


# --------------------------------------------------------------------
# Tamil TTS runs in its own OS process (src/indic_tts_worker.py), not
# in-process. This isolation was originally built because the old engine
# (indic-parler-tts) shared a small 4GB GPU with faster-whisper and could
# segfault the entire process outright -- this project's chat_web.py
# server died silently (no Python traceback, just gone) while a Tamil
# TTS generation was running. The current engine (AI4Bharat's
# FastPitch+HiFiGAN, see indic_tts.py) runs on CPU and no longer has that
# specific GPU-contention crash risk, but the isolation is kept anyway:
# it's still a large foreign dependency tree (coqui-tts, numba, etc.)
# with native extensions, and a try/except still can't catch a native
# crash in-process -- a separate worker means one can never take down
# Whisper or English/Hindi TTS with it. The worker loads the model once
# and stays warm across calls; only a crash or the timeout below
# triggers a respawn.
_indic_worker_process: subprocess.Popen | None = None
_indic_worker_lock = threading.Lock()
# Measured live on this project's own dev machine: a genuinely cold
# worker -- a brand new OS process paying Python startup + import (torch,
# coqui-tts, numba JIT) + model load from disk, all from scratch -- took
# ~35s; every request after that (same warm worker) took ~1.5-2s per
# sentence. Both dramatically faster than the old indic-parler-tts
# engine's 20-150s, but the cold case is meaningfully slower than
# reusing an already-warm Python process for the same load would
# suggest -- an earlier version of this timeout (60s) and index.html's
# client-side TTS_TIMEOUT_MS_TA (30s) were both based on that
# warm-process number and cut it too close live. This timeout also
# covers a from-scratch checkpoint download on a slow connection on a
# machine's very first-ever Tamil request (chat_web.py also kicks that
# download off proactively at startup so it's usually already done by
# the time a real request needs it -- see predownload_in_background()).
_INDIC_WORKER_TIMEOUT_S = 90


def _indic_worker_script_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "indic_tts_worker.py")


def _spawn_indic_worker() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, _indic_worker_script_path()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered: matches the worker's one-JSON-object-per-line protocol
    )


def _kill_indic_worker() -> None:
    global _indic_worker_process
    if _indic_worker_process is not None:
        try:
            _indic_worker_process.kill()
        except Exception:  # noqa: BLE001 -- already dead is fine, that's what we wanted anyway
            pass
        _indic_worker_process = None


atexit.register(_kill_indic_worker)  # don't leave an orphaned GPU-holding process behind on exit


def _readline_with_timeout(stream, timeout_s: float) -> str | None:
    """readline() with a timeout, since pipes don't support select() on
    Windows. Returns None on timeout OR on a clean EOF (worker exited) --
    both mean "no response came back", and the caller treats them the
    same way (check the process's exit code to tell them apart).
    """
    result: queue.Queue = queue.Queue(maxsize=1)

    def _reader():
        try:
            result.put(stream.readline())
        except Exception:  # noqa: BLE001 -- a broken pipe mid-read is just "no response"
            result.put("")

    threading.Thread(target=_reader, daemon=True).start()
    try:
        line = result.get(timeout=timeout_s)
    except queue.Empty:
        return None
    return line or None


def synthesize_tamil_via_worker(text: str, out_path: str) -> str:
    """Sends one synthesis request to the persistent Tamil TTS worker
    process, spawning it first if it isn't running yet (or respawning it
    if a previous call left it dead/hung). Raises RuntimeError with a
    clear reason on any failure -- crash, timeout, or the worker's own
    reported error -- callers already treat TTS failures this way (see
    chat_web.py's /api/tts try/except).
    """
    global _indic_worker_process
    with _indic_worker_lock:
        if _indic_worker_process is None or _indic_worker_process.poll() is not None:
            _indic_worker_process = _spawn_indic_worker()

        worker = _indic_worker_process
        if worker.stdin is None or worker.stdout is None:
            _kill_indic_worker()
            raise RuntimeError("Tamil TTS worker process was spawned without stdio pipes.")

        try:
            worker.stdin.write(json.dumps({"text": text, "out_path": out_path}) + "\n")
            worker.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            _kill_indic_worker()
            raise RuntimeError(f"Tamil TTS worker process is unavailable: {exc}") from exc

        response_line = _readline_with_timeout(worker.stdout, _INDIC_WORKER_TIMEOUT_S)
        if response_line is None:
            returncode = worker.poll()
            _kill_indic_worker()  # always respawn fresh next call, whether it crashed or just hung
            if returncode is not None:
                raise RuntimeError(f"Tamil TTS worker process crashed (exit code {returncode}).")
            raise RuntimeError(
                f"Tamil TTS worker process timed out after {_INDIC_WORKER_TIMEOUT_S}s."
            )

        response = json.loads(response_line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "Tamil TTS failed (no error detail given).")
        return out_path


# Cache of loaded PiperVoice objects, one per language -- mirrors indic_tts.
# py's _load_model() pattern: load once from disk, reuse forever. The old
# subprocess-per-call design paid Piper's ~60MB ONNX model load AND a fresh
# process spawn on every single synthesis; onnxruntime's InferenceSession.
# run() is documented safe for concurrent calls on one session, and this
# process already serves requests on multiple threads (Flask dev threaded=
# True / waitress's pool), so one cached voice per language can safely
# serve overlapping requests with no extra lock -- consistent with _gpu_lock
# in chat_web.py never having covered Piper either (it's CPU-only).
_piper_voices: dict[str, "PiperVoice"] = {}
_piper_voices_lock = threading.Lock()


def _load_piper_voice(language: str):
    """Loads (and caches) the in-process PiperVoice for `language`. See the
    _piper_voices comment above for why this replaced a subprocess call.
    """
    if language in _piper_voices:
        return _piper_voices[language]
    with _piper_voices_lock:
        if language not in _piper_voices:  # re-check: another thread may have loaded it first
            from piper.voice import PiperVoice

            voice_path = PIPER_VOICES.get(language, PIPER_VOICES["en"])
            _piper_voices[language] = PiperVoice.load(voice_path)
    return _piper_voices[language]


def synthesize_speech(
    text: str, out_path: str = os.path.join(LOG_DIR, "_tts_out.wav"), language: str = "en"
) -> str:
    """Synthesizes speech for `text` in the given language and writes a
    wav file to `out_path`. English and Hindi go through an in-process,
    cached `piper.voice.PiperVoice` (see _load_piper_voice) -- no subprocess,
    no network call involved; Tamil has no Piper voice, so it's routed to
    the separate `indic_tts` module instead (see CONFIG.tamil_tts_enabled).
    `text` is truncated to MAX_SPEECH_CHARS (MAX_SPEECH_CHARS_TA for
    Tamil -- see truncate_for_speech) -- the caller's full, untruncated
    text is still whatever gets displayed/logged; only what actually gets
    synthesized is capped. Callers doing their own sentence-chunking (see
    index.html) already pass one sentence at a time in the common case, so
    this cap rarely trims anything real -- it's a safety net, not the
    primary length control.
    """
    text = truncate_for_speech(text, MAX_SPEECH_CHARS_TA if language == "ta" else MAX_SPEECH_CHARS)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if language == "ta":
        if not CONFIG.tamil_tts_enabled:
            raise RuntimeError(
                "Tamil speech synthesis is disabled (TAMIL_TTS_ENABLED=0). "
                "Remove that override, or set it back to 1, to re-enable it."
            )
        return synthesize_tamil_via_worker(text, out_path)

    import wave

    voice = _load_piper_voice(language)
    with wave.open(out_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
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
            text, conf_result, _detected_language = transcribe_with_confidence(
                whisper_model, processed_audio
            )
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
