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

    # --- confidence gating (confidence_check.py) ---
    # See that module's docstring for how to re-tune this against your own
    # logs/confidence_log.csv.
    confidence_threshold: float = -0.6
    confidence_log_path: str = os.path.join("logs", "confidence_log.csv")

    # --- LLM (Ollama) ---
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_s: int = 180          # covers Ollama's one-time cold-load
    ollama_max_retries: int = 2          # retries on transient connection errors

    # --- TTS (Piper) ---
    piper_executable: str = field(default_factory=_default_piper_executable)
    piper_model_path: str = os.path.join("models", "piper", "en_US-lessac-medium.onnx")

    # --- logging ---
    log_dir: str = "logs"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 3

    # --- web servers (chat_web.py, record_web.py) ---
    chat_web_port: int = 5006
    record_web_port: int = 5005
    max_upload_bytes: int = 25 * 1024 * 1024  # 25MB cap on browser audio uploads
    production: bool = False             # PRODUCTION=1 -> serve via waitress

    @property
    def pipeline_log_path(self) -> str:
        return os.path.join(self.log_dir, "pipeline.log")

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
            ollama_url=os.environ.get("OLLAMA_URL", defaults.ollama_url),
            ollama_model=os.environ.get("OLLAMA_MODEL", defaults.ollama_model),
            piper_executable=os.environ.get("PIPER_EXECUTABLE", _default_piper_executable()),
            piper_model_path=os.environ.get("PIPER_MODEL_PATH", defaults.piper_model_path),
            log_dir=os.environ.get("LOG_DIR", defaults.log_dir),
            chat_web_port=_env_int("CHAT_WEB_PORT", defaults.chat_web_port),
            record_web_port=_env_int("RECORD_WEB_PORT", defaults.record_web_port),
            production=_env_bool("PRODUCTION", defaults.production),
        )


CONFIG = Config.from_env()
