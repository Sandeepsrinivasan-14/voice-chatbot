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
import logging
import os
import sys
import threading
import time
import uuid

import requests
from flask import Flask, Response, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from browser_audio import decode_browser_audio  # noqa: E402
from config import CONFIG  # noqa: E402
import pipeline  # noqa: E402  (also wires up cuda_dlls before faster_whisper loads)
from confidence_check import Decision  # noqa: E402
from preprocessing import normalize_audio  # noqa: E402
from web_common import register_error_handlers, run_app  # noqa: E402

try:
    # The continuous-learning layer (see src/learning/) is entirely
    # optional -- CONFIG.learning_enabled defaults to False, and even when
    # it's on, every call into this package below is wrapped in try/except
    # so a bug here can never break /api/transcribe's core response. This
    # import guard is the outermost layer of that: if the package itself
    # fails to import for some reason, the rest of chat_web.py still works
    # exactly as it did before this feature existed.
    from learning import audio_cache, correction_engine, correction_memory, feedback_store  # noqa: E402
    from learning import model_registry, scheduler as learning_scheduler  # noqa: E402
    from learning.profiles import resolve_profile_id, resolve_session_id  # noqa: E402

    _LEARNING_AVAILABLE = True
except Exception:  # noqa: BLE001 -- see comment above
    _LEARNING_AVAILABLE = False

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_chat")
MAX_HISTORY_MESSAGES = 12  # ~6 exchanges of context sent back to Ollama each turn
LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "ta": "Tamil"}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = CONFIG.max_upload_bytes
register_error_handlers(app)
_whisper_model = None
_whisper_device = None
_whisper_model_version = "stock"  # registry version string, or "stock" for CONFIG.whisper_model_size
_whisper_swap_lock = threading.Lock()  # guards swapping the globals themselves, not their use

# Serializes every GPU-using call this process makes (Whisper transcription
# AND Tamil TTS -- Piper/Hindi are CPU-side subprocess calls with no GPU
# involvement, so they don't need this). Found the hard way in live testing:
# this project's dev machine has a 4GB GPU, and letting Whisper and
# indic_tts (Tamil's TTS engine) hit CUDA concurrently -- e.g. one browser
# tab transcribing while another turn's reply is still being synthesized --
# corrupted the CUDA context enough to fail an unrelated Whisper call with
# `cudaErrorInvalidDevice: invalid device ordinal`, and made the *next*
# transcription noticeably slower even after it recovered. Flask's dev
# server (and waitress in production, see run_app) both serve requests on
# multiple threads, so without this lock those two GPU users really can
# run at the same instant.
_gpu_lock = threading.Lock()


def get_whisper_model():
    """Loaded once and cached. `__main__` below calls this eagerly at
    server startup so the load-time cost lands before anyone connects,
    not on someone's first spoken message. Still lazy with respect to
    plain module import (e.g. from tests), which never triggers a load.

    Callers should grab the model via this function once per request and
    use that local reference for the whole request -- swap_whisper_model()
    below only ever reassigns the module globals, it never mutates a
    WhisperModel object in place, so an in-flight request holding an old
    reference keeps working unaffected by a concurrent swap.
    """
    global _whisper_model, _whisper_device
    if _whisper_model is None:
        with _whisper_swap_lock:
            if _whisper_model is None:
                _load_initial_whisper_model()
    return _whisper_model


def _load_initial_whisper_model() -> None:
    """Loads the model to actually start serving with: the current
    production model from the learning registry if one exists and
    learning is enabled, otherwise the stock CONFIG.whisper_model_size --
    exactly mirroring what a fresh `python -m learning.training --promote`
    followed by a manual restart used to require, minus the restart.
    """
    global _whisper_model, _whisper_device, _whisper_model_version
    production = None
    if _LEARNING_AVAILABLE and CONFIG.learning_enabled:
        try:
            production = model_registry.get_production_model(CONFIG.learning_registry_path)
        except Exception:  # noqa: BLE001 -- registry problems must never block startup
            logging.getLogger("chat_web").exception(
                "Could not read learning model registry; starting with the stock model."
            )
    if production is not None:
        try:
            model, device = pipeline.load_whisper_model_from_path(
                production["artifact_path"]
            )
            _whisper_model, _whisper_device = model, device
            _whisper_model_version = production["version"]
            print(
                f"[chat_web] faster-whisper ready on {device} "
                f"(production model {production['version']}, from continuous learning)"
            )
            return
        except Exception:  # noqa: BLE001 -- fall through to the stock model on any failure
            logging.getLogger("chat_web").exception(
                "Failed to load promoted model %s; falling back to the stock model.",
                production.get("version"),
            )

    _whisper_model, _whisper_device = pipeline.load_whisper_model()
    _whisper_model_version = "stock"
    print(f"[chat_web] faster-whisper ready on {_whisper_device}")


def swap_whisper_model(version: str, artifact_path: str) -> None:
    """Hot-swap the live model onto a newly promoted candidate, without
    interrupting any transcription already in progress. Loads the new
    model fully (including its own GPU/CPU fallback + smoke test) *before*
    touching any global, so a bad candidate never leaves the server
    without a working model; only reassigns the globals -- a single
    atomic-under-the-GIL rebind -- once the new model is confirmed loadable.
    Called by the learning scheduler's on_promoted callback; safe to call
    from a background thread.
    """
    global _whisper_model, _whisper_device, _whisper_model_version
    logger = logging.getLogger("chat_web")
    logger.info("Loading newly promoted model %s for hot-swap...", version)
    new_model, new_device = pipeline.load_whisper_model_from_path(artifact_path)
    with _whisper_swap_lock:
        _whisper_model, _whisper_device, _whisper_model_version = new_model, new_device, version
    logger.info("Live chatbot now serving model %s (device=%s).", version, new_device)


def get_whisper_model_version() -> str:
    return _whisper_model_version


# --------------------------------------------------------------------
# Per-language LLM routing: live testing (see project notes) found no
# single small/medium general-purpose Ollama model replies coherently in
# Tamil consistently, regardless of size -- but a Tamil-specialized model
# does, at the cost of breaking on Hindi. So Tamil routes to its own
# model; English/Hindi keep using CONFIG.ollama_model unchanged.
# --------------------------------------------------------------------
_ollama_ta_model_ready = False  # set True once confirmed pulled (or pull disabled)


def get_ollama_model_for_language(language: str) -> str:
    if language == "ta" and CONFIG.ollama_model_ta and _ollama_ta_model_ready:
        return CONFIG.ollama_model_ta
    return CONFIG.ollama_model


def _ollama_has_model(model_name: str) -> bool:
    try:
        resp = requests.get(f"{pipeline.OLLAMA_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        names = {m.get("name") for m in resp.json().get("models", [])}
        # Ollama lists pulled models with an explicit tag (defaulting to
        # ":latest" when none was requested) even though a bare name like
        # "org/model" with no ":tag" is exactly what both `ollama pull` and
        # this project's config accept and mean the same thing.
        return model_name in names or f"{model_name}:latest" in names
    except Exception:  # noqa: BLE001 -- Ollama not reachable yet at startup is not fatal
        return False


def ensure_ollama_ta_model_pulled() -> None:
    """Pulls CONFIG.ollama_model_ta via Ollama's own /api/pull if it isn't
    already present, so a fresh install needs no manual `ollama pull` step
    for Tamil to work well -- mirrors indic_tts.py's self-installing
    dependencies. Runs on a background daemon thread from __main__ below:
    a multi-GB pull must never block the server from accepting English/
    Hindi requests (or Tamil ones -- they just fall back to
    CONFIG.ollama_model until this finishes). Marks the model ready only
    on confirmed success; any failure (offline, disk full, registry
    hiccup) leaves Tamil silently using the fallback model rather than
    breaking the chatbot.
    """
    global _ollama_ta_model_ready
    logger = logging.getLogger("chat_web")
    if not CONFIG.ollama_model_ta:
        return
    if _ollama_has_model(CONFIG.ollama_model_ta):
        _ollama_ta_model_ready = True
        logger.info("Tamil LLM model %s already present.", CONFIG.ollama_model_ta)
        return
    if not CONFIG.ollama_auto_pull_ta_model:
        logger.info(
            "Tamil LLM model %s not present and auto-pull is disabled; "
            "Tamil replies will use %s instead.",
            CONFIG.ollama_model_ta, CONFIG.ollama_model,
        )
        return

    logger.info(
        "Pulling Tamil LLM model %s in the background (multi-GB, one-time; "
        "Tamil replies use %s until this finishes)...",
        CONFIG.ollama_model_ta, CONFIG.ollama_model,
    )
    try:
        with requests.post(
            f"{pipeline.OLLAMA_URL}/api/pull",
            json={"name": CONFIG.ollama_model_ta, "stream": True},
            stream=True,
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                status = json.loads(line).get("status", "")
                if status and status != "success":
                    logger.debug("Pulling %s: %s", CONFIG.ollama_model_ta, status)
        if _ollama_has_model(CONFIG.ollama_model_ta):
            _ollama_ta_model_ready = True
            logger.info("Tamil LLM model %s pulled successfully.", CONFIG.ollama_model_ta)
        else:
            logger.warning(
                "Pull of %s finished without error but the model isn't listed; "
                "Tamil will keep using %s.", CONFIG.ollama_model_ta, CONFIG.ollama_model,
            )
    except Exception:  # noqa: BLE001 -- must never crash the server or block startup
        logger.exception(
            "Failed to auto-pull Tamil LLM model %s; Tamil replies will use %s instead.",
            CONFIG.ollama_model_ta, CONFIG.ollama_model,
        )


def ensure_tamil_tts_checkpoint_pulled() -> None:
    """Downloads indic_tts.py's ~1.5GB FastPitch+HiFiGAN checkpoint if it
    isn't already present, so a fresh install needs no manual step for
    Tamil TTS to work well -- same rationale as
    ensure_ollama_ta_model_pulled() above. Runs on a background daemon
    thread from __main__ below: the first real Tamil TTS request would
    otherwise have to pay this download itself (see indic_tts.py's own
    lazy call to the same function), which on a slow connection could
    make a normal request look like a hang.
    """
    logger = logging.getLogger("chat_web")
    try:
        from indic_tts import ensure_checkpoint_downloaded

        ensure_checkpoint_downloaded()
    except Exception:  # noqa: BLE001 -- must never crash the server or block startup
        logger.exception(
            "Failed to pre-download the Tamil TTS checkpoint; the first real Tamil "
            "TTS request will download it instead (and take longer than usual)."
        )


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/health")
def health():
    return jsonify(
        {"ok": True, "whisper_device": _whisper_device, "whisper_model_version": _whisper_model_version}
    )


@app.route("/api/learning/status")
def learning_status():
    """Read-only status of the automated continuous-learning loop: what
    the scheduler is doing, when it last ran/trained/promoted, and which
    model version the live chatbot is actually serving right now. 404s
    when learning is disabled entirely, matching this project's existing
    convention that a disabled feature acts as if it doesn't exist.
    """
    if not (_LEARNING_AVAILABLE and CONFIG.learning_enabled):
        return jsonify({"error": "Continuous learning is not enabled."}), 404
    status = learning_scheduler.get_status(
        CONFIG.learning_scheduler_state_path, CONFIG.learning_registry_path
    )
    status["live_model_version"] = _whisper_model_version
    status["live_model_device"] = _whisper_device
    status["auto_training_enabled"] = CONFIG.learning_auto_training_enabled
    return jsonify(status)


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

    # "auto" (default) lets faster-whisper detect English/Hindi/Tamil itself;
    # the UI can send an explicit code to override detection on a clip
    # that's too short/ambiguous to auto-detect reliably.
    requested_language = (request.form.get("language") or "auto").strip().lower()

    model = get_whisper_model()
    with _gpu_lock:
        text, conf_result, detected_language = pipeline.transcribe_with_confidence(
            model, processed_audio, language=requested_language
        )

    response = {
        "text": text,
        "decision": conf_result.decision.value,
        "avg_logprob": conf_result.avg_logprob,
        "threshold": conf_result.threshold,
        "accepted": conf_result.decision == Decision.ACCEPTED,
        "whisper_device": _whisper_device,
        "language": detected_language,
    }

    # Everything below is additive and only runs when the feature is on --
    # with CONFIG.learning_enabled at its default (False), `response` above
    # is returned completely unchanged, key-for-key, from before this
    # feature existed.
    if CONFIG.learning_enabled and _LEARNING_AVAILABLE and response["accepted"]:
        try:
            response.update(
                _apply_personalized_correction(
                    text, conf_result.avg_logprob, request.form, processed_audio
                )
            )
        except Exception as exc:  # noqa: BLE001 -- learning bugs must never break transcription
            app.logger.warning("Learning layer raised during transcribe (%s); ignoring.", exc)

    return jsonify(response)


def _apply_personalized_correction(text: str, avg_logprob: float, form, processed_audio) -> dict:
    """Looks up this profile's learned corrections, scores them, and
    returns the extra /api/transcribe response fields: event_id and
    profile_id always; text/original_text/auto_corrected only if something
    was auto-applied; correction_suggestions only if there's a
    medium-confidence match. Pure w.r.t. the database -- never writes a
    feedback event, only reads correction_rules; feedback_endpoint() below
    is the only write path. event_id is a fresh correlation id (not a
    stored row) the client echoes back if it later submits a correction for
    this transcription.
    """
    event_id = str(uuid.uuid4())
    profile_id = resolve_profile_id(form.get("profile_id"))
    extra: dict = {
        "event_id": event_id,
        "profile_id": profile_id,
        "session_id": resolve_session_id(form.get("session_id")),
    }

    if CONFIG.learning_persist_audio_enabled:
        # Held in memory only; written to disk only if this specific
        # transcription is later corrected (see feedback_endpoint()).
        audio_cache.stash(event_id, processed_audio, pipeline.SAMPLE_RATE)

    if not CONFIG.learning_personalized_correction_enabled:
        return extra

    suggestions = correction_engine.suggest_corrections(
        CONFIG.learning_db_path,
        profile_id,
        text,
        avg_logprob=avg_logprob,
        suggest_threshold=CONFIG.learning_suggest_threshold,
        auto_apply_threshold=CONFIG.learning_auto_apply_threshold,
        frequency_saturation=CONFIG.learning_rule_frequency_saturation,
        recency_halflife_days=CONFIG.learning_rule_recency_halflife_days,
        context_aware_enabled=CONFIG.learning_context_aware_enabled,
        allow_auto_apply=CONFIG.learning_auto_apply_enabled,
    )
    corrected_text, applied = correction_engine.apply_auto_corrections(text, suggestions)
    if applied:
        extra["text"] = corrected_text
        extra["original_text"] = text
        extra["auto_corrected"] = True

    suggest_band = [s for s in suggestions if s.band == "suggest"]
    if suggest_band:
        extra["correction_suggestions"] = [
            {
                "rule_id": s.rule_id,
                "original_phrase": s.original_phrase,
                "corrected_phrase": s.corrected_phrase,
                "score": s.score,
            }
            for s in suggest_band
        ]
    return extra


@app.route("/api/feedback", methods=["POST"])
def feedback_endpoint():
    """Records a user's correction of a transcript: original and corrected
    text stored as separate, immutable fields (see
    learning/feedback_store.py -- the DB itself rejects UPDATE/DELETE on
    that table). This never re-sends anything to Ollama -- the chat turn
    that used the original transcript already happened; a correction here
    is purely about improving future transcription.
    """
    if not (CONFIG.learning_enabled and CONFIG.learning_feedback_enabled):
        return jsonify({"error": "Feedback collection is disabled."}), 400
    if not _LEARNING_AVAILABLE:
        return jsonify({"error": "Learning layer unavailable."}), 500

    data = request.get_json(force=True) or {}
    original_text = (data.get("original_text") or "").strip()
    corrected_text = (data.get("corrected_text") or "").strip()
    if not original_text or not corrected_text:
        return jsonify({"error": "original_text and corrected_text are required."}), 400
    if original_text == corrected_text:
        return jsonify({"error": "corrected_text must differ from original_text."}), 400

    profile_id = resolve_profile_id(data.get("profile_id"))
    session_id = resolve_session_id(data.get("session_id"))
    avg_logprob = data.get("avg_logprob")
    stt_confidence = float(avg_logprob) if isinstance(avg_logprob, (int, float)) else None

    # If this transcription's audio is still cached (persistence was on and
    # the correction arrived before the cache entry aged out), write it to
    # permanent storage now -- this is the only moment audio is ever
    # written to disk in this feature. transcribe_event_id is the
    # correlation id /api/transcribe handed back as "event_id"; it is not
    # the feedback row's own id.
    audio_ref = None
    transcribe_event_id = data.get("event_id")
    if CONFIG.learning_persist_audio_enabled and transcribe_event_id:
        try:
            audio_ref = audio_cache.persist(
                str(transcribe_event_id), CONFIG.learning_audio_dir, str(transcribe_event_id)
            )
        except Exception as exc:  # noqa: BLE001 -- a failed audio write must not lose the correction
            app.logger.warning("Failed to persist audio for a feedback event (%s).", exc)

    try:
        event = feedback_store.insert_feedback_event(
            CONFIG.learning_db_path,
            profile_id=profile_id,
            session_id=session_id,
            original_text=original_text,
            corrected_text=corrected_text,
            context_before=(data.get("context_before") or None),
            context_after=(data.get("context_after") or None),
            audio_ref=audio_ref,
            language=data.get("language") or "en",
            stt_confidence=stt_confidence,
            stt_model_size=CONFIG.whisper_model_size,
            stt_device=_whisper_device,
        )
        if CONFIG.learning_personalized_correction_enabled:
            correction_memory.upsert_rule_from_event(CONFIG.learning_db_path, event)
    except Exception as exc:
        app.logger.warning("Failed to store feedback event (%s).", exc)
        return jsonify({"error": "Could not store feedback."}), 500

    return jsonify({
        "ok": True,
        "event_id": event.event_uuid,
        "diff": [
            {"op": d.op, "original_phrase": d.original_phrase, "corrected_phrase": d.corrected_phrase}
            for d in event.diffs
        ],
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
    language = (data.get("language") or "en").strip().lower()
    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    system_content = pipeline.SYSTEM_PROMPT
    if language in ("hi", "ta"):
        # A per-turn nudge on top of SYSTEM_PROMPT's general instruction --
        # helps on short/ambiguous messages where the language is otherwise
        # only implicit in the transcript's own script.
        system_content += (
            f" The user's message is in {LANGUAGE_NAMES[language]}; "
            f"reply in {LANGUAGE_NAMES[language]}, not English."
        )

    messages = [{"role": "system", "content": system_content}]
    messages.extend(history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": user_message})

    ollama_model = get_ollama_model_for_language(language)

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
                    json={"model": ollama_model, "messages": messages, "stream": True},
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
    language = (data.get("language") or "en").strip().lower()
    if not text:
        return jsonify({"error": "Empty text."}), 400
    # A unique path per request: pipeline.synthesize_speech()'s own default
    # is a single shared file, fine for the CLI's one-request-at-a-time
    # loop but not safe here -- under waitress (PRODUCTION=1) requests are
    # served on multiple threads, and two concurrent /api/tts calls
    # writing the same path could serve one tab the other's audio, or hit
    # a Windows file-lock error.
    wav_path = os.path.join(CONFIG.log_dir, f"_tts_out_{uuid.uuid4().hex}.wav")
    try:
        # Neither branch touches the GPU: Piper (English/Hindi) is a
        # CPU-side subprocess, and Tamil's FastPitch+HiFiGAN engine
        # (indic_tts.py) deliberately runs on CPU too (see its module
        # docstring) -- so _gpu_lock, which only ever needed to guard
        # Whisper against GPU contention, doesn't need to wrap either.
        pipeline.synthesize_speech(text, out_path=wav_path, language=language)
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        return Response(wav_bytes, mimetype="audio/wav")
    except Exception as exc:
        return jsonify({"error": f"TTS failed: {exc}"}), 500
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


if __name__ == "__main__":
    print("Loading faster-whisper now, before accepting connections -- this")
    print("takes a while on first run (larger models take longer) but means")
    print("your first spoken message won't have to wait for it.")
    print("Everything here stays on this machine -- close the tab/Ctrl+C to stop.")
    get_whisper_model()

    if CONFIG.ollama_model_ta and CONFIG.ollama_auto_pull_ta_model:
        # Non-blocking: Tamil replies use CONFIG.ollama_model as a fallback
        # until this finishes (see get_ollama_model_for_language()), so a
        # multi-GB pull never delays the server accepting requests.
        threading.Thread(
            target=ensure_ollama_ta_model_pulled, name="ollama-ta-model-pull", daemon=True
        ).start()

    if CONFIG.tamil_tts_enabled:
        # Non-blocking, same rationale as the Ollama pull above -- Tamil
        # TTS just downloads it itself on first real use if this hasn't
        # finished yet (see indic_tts.py's own call to the same function).
        threading.Thread(
            target=ensure_tamil_tts_checkpoint_pulled, name="tamil-tts-checkpoint-pull", daemon=True
        ).start()

    if _LEARNING_AVAILABLE and CONFIG.learning_enabled and CONFIG.learning_auto_training_enabled:
        # Runs the check-and-maybe-train loop on a background daemon thread
        # for the lifetime of this process. Fully inert (start_background_
        # scheduler returns None without starting anything) unless both
        # flags above are set -- see src/learning/scheduler.py.
        learning_scheduler.start_background_scheduler(CONFIG, on_promoted=swap_whisper_model)
        print(
            f"[chat_web] automated learning scheduler running "
            f"(every {CONFIG.learning_training_interval_minutes}min, "
            f"needs >={CONFIG.learning_min_new_events_for_training} new corrections)."
        )

    run_app(app, host="127.0.0.1", port=CONFIG.chat_web_port, name="chat_web")
