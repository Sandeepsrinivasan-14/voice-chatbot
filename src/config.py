"""
Centralized, validated configuration for the whole project.

Every tunable that used to be a scattered `os.environ.get(NAME, default)`
call duplicated across pipeline.py, chat_web.py, record_web.py,
verify_setup.py, and confidence_check.py now lives here, in one place,
read and parsed once. A local `.env` file (see `.env.example`) is picked
up automatically via python-dotenv, so you don't need to `set FOO=bar` in
every new terminal.

This is a refactor of *where* config lives, not *how* you set it -- every
environment variable name is unchanged from before (WHISPER_MODEL_SIZE,
OLLAMA_URL, OLLAMA_MODEL, PIPER_EXECUTABLE, PIPER_MODEL_PATH,
CHAT_WEB_PORT, RECORD_WEB_PORT, ...).

Usage:
    from config import CONFIG
    print(CONFIG.ollama_model)

Tests should NOT import the module-level `CONFIG` singleton when they want
an isolated configuration -- construct one directly instead:
    Config.from_env()               # reads the real environment
    Config()                        # pure defaults, ignores environment
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()  # no-op if there's no .env file in the working directory


def _default_piper_executable() -> str:
    """piper-tts installs a `piper`/`piper.exe` console script alongside
    the Python interpreter it was pip-installed into (venv/Scripts on
    Windows, venv/bin on Linux/Mac). Resolving relative to sys.executable
    means it works whether or not the venv was `activate`d in this shell.
    """
    candidate = os.path.join(
        os.path.dirname(sys.executable), "piper.exe" if os.name == "nt" else "piper"
    )
    return candidate if os.path.exists(candidate) else "piper"


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Environment variable {name}={val!r} is not a valid number") from None


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Environment variable {name}={val!r} is not a valid integer") from None


@dataclass(frozen=True)
class Config:
    # --- audio / capture (pipeline.py's live mic loop) ---
    sample_rate: int = 16000            # required by both webrtcvad and whisper
    frame_ms: int = 30
    vad_aggressiveness: int = 2         # 0 (permissive) .. 3 (strict)
    max_record_seconds: int = 15

    # --- STT (faster-whisper) ---
    # "small" fits comfortably on a 4GB GPU alongside a small quantized
    # Ollama model. int8_float16 halves VRAM vs float16 with a negligible
    # accuracy hit -- worth it when VRAM is the binding constraint.
    whisper_model_size: str = "small"
    whisper_compute_type_gpu: str = "int8_float16"
    whisper_compute_type_cpu: str = "int8"
    beam_size_fast: int = 1             # first-pass greedy-ish decode, fast
    beam_size_retry: int = 5            # wider search, only on low confidence
    # "auto" lets faster-whisper detect the spoken language per utterance
    # (needed for Hindi/Tamil support); set to "en"/"hi"/"ta" to force one
    # instead of relying on detection (useful for short/ambiguous clips).
    whisper_language: str = "auto"

    # --- confidence gating (confidence_check.py) ---
    # See that module's docstring for how to re-tune this against your own
    # logs/confidence_log.csv.
    confidence_threshold: float = -0.6
    confidence_log_path: str = os.path.join("logs", "confidence_log.csv")

    # --- LLM (Ollama) ---
    ollama_url: str = "http://localhost:11434"
    # Single model for all three languages. Earlier testing (llama3.2:3b,
    # qwen2.5:3b/7b-instruct) found no small general-purpose model replies
    # coherently in Tamil, which led to a dual-model setup (a Tamil-specialized
    # 7B model routed only for Tamil). gemma3n:e2b replaced that: live-tested
    # on 5 Tamil prompts + Hindi + English, it was coherent and on-topic in
    # all three (4/5 Tamil answers correct; one hallucinated a word mix-up
    # between "weather"/"vanilla" -- acceptable, no worse than other models'
    # occasional misses), runs 100% on GPU using ~1.7GB VRAM, and responds in
    # well under 2s once warm. This removes the GPU-lock contention that
    # existed between Whisper STT and the old Tamil-only model's TTS/LLM
    # calls, since there's now only one model to keep resident.
    ollama_model: str = "gemma3n:e2b"
    ollama_timeout_s: int = 180          # covers Ollama's one-time cold-load
    ollama_max_retries: int = 2          # retries on transient connection errors
    # Tamil-specific model override -- empty string means "no override, use
    # ollama_model for Tamil too". Left in place (rather than removed) in
    # case a future language needs this same escape hatch again.
    ollama_model_ta: str = ""
    # chat_web.py pulls ollama_model_ta itself in the background at startup
    # if it isn't already present (see ensure_ollama_model_pulled()) so no
    # manual `ollama pull` step is needed -- set this False to disable that
    # background pull (e.g. a machine that deliberately wants to skip the
    # ~4GB download and just fall back to ollama_model for Tamil too).
    ollama_auto_pull_ta_model: bool = True

    # --- TTS (Piper) ---
    piper_executable: str = field(default_factory=_default_piper_executable)
    piper_model_path: str = os.path.join("models", "piper", "en_US-lessac-medium.onnx")
    # Hindi voice, same Piper binary/invocation as English -- just a
    # second .onnx file. Piper ships no Tamil voice at all (see below).
    piper_model_path_hi: str = os.path.join("models", "piper", "hi_IN-pratham-medium.onnx")

    # --- TTS (Tamil, via AI4Bharat FastPitch+HiFiGAN) ---
    # Piper has no Tamil voice, so Tamil speech uses a separate engine
    # instead of Piper's tiny CLI binary -- AI4Bharat's FastPitch+HiFiGAN
    # checkpoint (their "Indic-TTS" project), run through coqui-tts's
    # Synthesizer. Replaced indic-parler-tts here after live comparison
    # found FastPitch+HiFiGAN both far faster (~1.5s/sentence on CPU vs.
    # 20-150s on GPU for parler-tts) and more accurate (near-perfect
    # Whisper round-trip vs. real word-level errors for parler-tts on the
    # same test sentence) -- see indic_tts.py's module docstring for the
    # full story. On by default -- the client requirement is that Tamil/
    # Hindi "just work" with no manual setup step, so indic_tts.py
    # auto-installs its own missing dependencies AND downloads the model
    # checkpoint on first real use (see that module's _load_model())
    # rather than requiring a human to run `pip install -r
    # requirements-indic-tts.txt` first. Set this to 0 to opt back out
    # (e.g. a deployment that only ever needs English/Hindi and wants to
    # avoid the ~1.5GB download entirely).
    tamil_tts_enabled: bool = True

    # --- logging ---
    log_dir: str = "logs"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 3

    # --- web servers (chat_web.py, record_web.py) ---
    # Bind address. Loopback by default -- the apps have no authentication,
    # so only widen this (e.g. WEB_HOST=0.0.0.0 inside a container) when the
    # port is published to loopback or sits behind your own access control.
    web_host: str = "127.0.0.1"
    chat_web_port: int = 5006
    record_web_port: int = 5005
    max_upload_bytes: int = 25 * 1024 * 1024  # 25MB cap on browser audio uploads
    production: bool = False             # PRODUCTION=1 -> serve via waitress

    # --- continuous learning: STT correction feedback (src/learning/) ---
    # Master switch; everything below is inert while this is False, and
    # /api/transcribe's response is byte-for-byte what it was before this
    # feature existed. The three sub-flags let feedback collection,
    # personalized correction lookup, and context-aware scoring be turned
    # off independently of each other once the master switch is on. See
    # README "Continuous learning" section.
    learning_enabled: bool = False
    learning_feedback_enabled: bool = True
    learning_personalized_correction_enabled: bool = True
    learning_context_aware_enabled: bool = True
    # Auto-apply is the one knob that rewrites what gets sent to the LLM
    # without the user confirming it. Off by default even when the master
    # switch is on; auto-apply is also unreachable whenever
    # learning_context_aware_enabled is False, regardless of this flag.
    learning_auto_apply_enabled: bool = False
    learning_db_path: str = os.path.join("data", "learning", "feedback.db")
    # Weighted correction score (see correction_scoring.py) needed to
    # surface a suggestion / to auto-apply one. Auto-apply also requires
    # learning_auto_apply_enabled and learning_context_aware_enabled.
    learning_suggest_threshold: float = 0.6
    learning_auto_apply_threshold: float = 0.92
    # How many times a correction must recur before frequency stops adding
    # to its score (diminishing returns past this point).
    learning_rule_frequency_saturation: int = 5
    # Recency-decay half-life for a learned rule's score contribution.
    learning_rule_recency_halflife_days: float = 30.0
    # Off by default: no audio is written to disk for a transcription that
    # nobody corrects. When on, the processed audio for an accepted
    # transcription is held in memory (src/learning/audio_cache.py) and
    # only written to learning_audio_dir if the user actually submits a
    # correction for it via /api/feedback.
    learning_persist_audio_enabled: bool = False
    learning_audio_dir: str = os.path.join("data", "learning", "audio")

    # --- continuous learning: automated scheduled fine-tuning ---
    # Master switch for the background training loop. Requires
    # learning_enabled=True as well; everything below is inert otherwise.
    # Off by default -- this is a heavy, optional layer on top of the
    # (already-default-on-when-learning_enabled) immediate correction
    # mechanism, which is completely unaffected by any of these flags.
    learning_auto_training_enabled: bool = False
    # How often the scheduler wakes up to check whether enough new
    # corrected data has accumulated to justify a training run. A wakeup
    # that finds too little data does nothing and waits for the next one.
    learning_training_interval_minutes: int = 60
    # Minimum number of *new* feedback events (since the last training run
    # that actually attempted a dataset) required before a wakeup will
    # trigger a real training cycle.
    learning_min_new_events_for_training: int = 50
    learning_training_language: str = "en"
    learning_datasets_root: str = os.path.join("data", "learning", "datasets")
    learning_models_root: str = os.path.join("data", "learning", "models")
    learning_registry_path: str = os.path.join("data", "learning", "models", "registry.json")
    learning_scheduler_state_path: str = os.path.join("data", "learning", "scheduler_state.json")
    learning_scheduler_lock_path: str = os.path.join("data", "learning", "scheduler.lock")
    # A lock older than this is assumed to belong to a crashed/killed
    # process rather than a genuinely still-running job, and is reclaimed
    # rather than blocking the scheduler forever.
    learning_training_lock_stale_seconds: int = 6 * 3600
    learning_training_min_relative_improvement: float = 0.05
    learning_training_base_model: str = "openai/whisper-small"
    learning_training_baseline_model: str = "small"
    learning_training_device: str = "cpu"
    # Whether a successful promotion should also make the *live* chatbot
    # swap to the new model automatically (hot-reload, no restart). If
    # False, promotion still updates the registry but the running process
    # keeps serving whatever model it already loaded -- an escape hatch for
    # operators who want to review a promotion before it goes live.
    learning_auto_deploy_enabled: bool = True

    @property
    def pipeline_log_path(self) -> str:
        return os.path.join(self.log_dir, "pipeline.log")

    @property
    def piper_voices(self) -> dict:
        """Language -> Piper .onnx voice path, for languages Piper can
        actually speak (English, Hindi). Tamil is handled by a separate
        engine (see tamil_tts_enabled) since no Piper Tamil voice exists.
        """
        return {"en": self.piper_model_path, "hi": self.piper_model_path_hi}

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config by reading the current environment (+ any loaded
        .env file). Call this fresh rather than mutating the module-level
        CONFIG singleton -- e.g. in tests, after monkeypatching os.environ.
        """
        defaults = cls()
        return cls(
            sample_rate=_env_int("SAMPLE_RATE", defaults.sample_rate),
            whisper_model_size=os.environ.get("WHISPER_MODEL_SIZE", defaults.whisper_model_size),
            confidence_threshold=_env_float("CONFIDENCE_THRESHOLD", defaults.confidence_threshold),
            confidence_log_path=os.environ.get(
                "CONFIDENCE_LOG_PATH", defaults.confidence_log_path
            ),
            whisper_language=os.environ.get("WHISPER_LANGUAGE", defaults.whisper_language),
            ollama_url=os.environ.get("OLLAMA_URL", defaults.ollama_url),
            ollama_model=os.environ.get("OLLAMA_MODEL", defaults.ollama_model),
            ollama_model_ta=os.environ.get("OLLAMA_MODEL_TA", defaults.ollama_model_ta),
            ollama_auto_pull_ta_model=_env_bool(
                "OLLAMA_AUTO_PULL_TA_MODEL", defaults.ollama_auto_pull_ta_model
            ),
            piper_executable=os.environ.get("PIPER_EXECUTABLE", _default_piper_executable()),
            piper_model_path=os.environ.get("PIPER_MODEL_PATH", defaults.piper_model_path),
            piper_model_path_hi=os.environ.get(
                "PIPER_MODEL_PATH_HI", defaults.piper_model_path_hi
            ),
            tamil_tts_enabled=_env_bool("TAMIL_TTS_ENABLED", defaults.tamil_tts_enabled),
            log_dir=os.environ.get("LOG_DIR", defaults.log_dir),
            web_host=os.environ.get("WEB_HOST", defaults.web_host).strip() or defaults.web_host,
            chat_web_port=_env_int("CHAT_WEB_PORT", defaults.chat_web_port),
            record_web_port=_env_int("RECORD_WEB_PORT", defaults.record_web_port),
            production=_env_bool("PRODUCTION", defaults.production),
            learning_enabled=_env_bool("LEARNING_ENABLED", defaults.learning_enabled),
            learning_feedback_enabled=_env_bool(
                "LEARNING_FEEDBACK_ENABLED", defaults.learning_feedback_enabled
            ),
            learning_personalized_correction_enabled=_env_bool(
                "LEARNING_PERSONALIZED_CORRECTION_ENABLED",
                defaults.learning_personalized_correction_enabled,
            ),
            learning_context_aware_enabled=_env_bool(
                "LEARNING_CONTEXT_AWARE_ENABLED", defaults.learning_context_aware_enabled
            ),
            learning_auto_apply_enabled=_env_bool(
                "LEARNING_AUTO_APPLY_ENABLED", defaults.learning_auto_apply_enabled
            ),
            learning_db_path=os.environ.get("LEARNING_DB_PATH", defaults.learning_db_path),
            learning_suggest_threshold=_env_float(
                "LEARNING_SUGGEST_THRESHOLD", defaults.learning_suggest_threshold
            ),
            learning_auto_apply_threshold=_env_float(
                "LEARNING_AUTO_APPLY_THRESHOLD", defaults.learning_auto_apply_threshold
            ),
            learning_rule_frequency_saturation=_env_int(
                "LEARNING_RULE_FREQUENCY_SATURATION", defaults.learning_rule_frequency_saturation
            ),
            learning_rule_recency_halflife_days=_env_float(
                "LEARNING_RULE_RECENCY_HALFLIFE_DAYS",
                defaults.learning_rule_recency_halflife_days,
            ),
            learning_persist_audio_enabled=_env_bool(
                "LEARNING_PERSIST_AUDIO_ENABLED", defaults.learning_persist_audio_enabled
            ),
            learning_audio_dir=os.environ.get("LEARNING_AUDIO_DIR", defaults.learning_audio_dir),
            learning_auto_training_enabled=_env_bool(
                "LEARNING_AUTO_TRAINING_ENABLED", defaults.learning_auto_training_enabled
            ),
            learning_training_interval_minutes=_env_int(
                "LEARNING_TRAINING_INTERVAL_MINUTES", defaults.learning_training_interval_minutes
            ),
            learning_min_new_events_for_training=_env_int(
                "LEARNING_MIN_NEW_EVENTS_FOR_TRAINING",
                defaults.learning_min_new_events_for_training,
            ),
            learning_training_language=os.environ.get(
                "LEARNING_TRAINING_LANGUAGE", defaults.learning_training_language
            ),
            learning_datasets_root=os.environ.get(
                "LEARNING_DATASETS_ROOT", defaults.learning_datasets_root
            ),
            learning_models_root=os.environ.get(
                "LEARNING_MODELS_ROOT", defaults.learning_models_root
            ),
            learning_registry_path=os.environ.get(
                "LEARNING_REGISTRY_PATH", defaults.learning_registry_path
            ),
            learning_scheduler_state_path=os.environ.get(
                "LEARNING_SCHEDULER_STATE_PATH", defaults.learning_scheduler_state_path
            ),
            learning_scheduler_lock_path=os.environ.get(
                "LEARNING_SCHEDULER_LOCK_PATH", defaults.learning_scheduler_lock_path
            ),
            learning_training_lock_stale_seconds=_env_int(
                "LEARNING_TRAINING_LOCK_STALE_SECONDS",
                defaults.learning_training_lock_stale_seconds,
            ),
            learning_training_min_relative_improvement=_env_float(
                "LEARNING_TRAINING_MIN_RELATIVE_IMPROVEMENT",
                defaults.learning_training_min_relative_improvement,
            ),
            learning_training_base_model=os.environ.get(
                "LEARNING_TRAINING_BASE_MODEL", defaults.learning_training_base_model
            ),
            learning_training_baseline_model=os.environ.get(
                "LEARNING_TRAINING_BASELINE_MODEL", defaults.learning_training_baseline_model
            ),
            learning_training_device=os.environ.get(
                "LEARNING_TRAINING_DEVICE", defaults.learning_training_device
            ),
            learning_auto_deploy_enabled=_env_bool(
                "LEARNING_AUTO_DEPLOY_ENABLED", defaults.learning_auto_deploy_enabled
            ),
        )


CONFIG = Config.from_env()
