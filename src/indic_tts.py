"""
Tamil TTS via AI4Bharat's FastPitch+HiFiGAN checkpoint (their "Indic-TTS"
project: https://github.com/AI4Bharat/Indic-TTS), run through coqui-tts's
`Synthesizer` class.

Replaced indic-parler-tts (a transformers/torch autoregressive model) after
a live comparison found FastPitch+HiFiGAN to be both far faster and more
accurate on this project's own test sentences:

  - Speed: ~1.5s per sentence on CPU, vs. 20-150s on GPU for parler-tts.
    FastPitch (acoustic model) and HiFiGAN (vocoder) are both small,
    non-autoregressive models -- the same architecture family as Piper,
    which is why Piper is already fast for English/Hindi. Parler-TTS is a
    large autoregressive transformer, which is *why* it was slow.
  - Accuracy: round-tripping synthesized audio back through Whisper STT
    came back near-perfect for FastPitch+HiFiGAN (only trivial
    spacing/nasal-sound differences), vs. genuine word-level errors for
    parler-tts on the same sentences (e.g. "இன்று" transcribed back as the
    wrong word "இஞ்சு").

Deliberately runs on CPU even when a GPU is present (`use_cuda=False`
below) -- the whole point of this swap is to free the small (4GB on this
project's own dev machine) GPU for Whisper STT + the LLM, since
FastPitch+HiFiGAN is fast enough on CPU that there's no reason to fight
over GPU memory the way parler-tts did (see pipeline.py's worker-isolation
comment for that history).

Piper (this project's default TTS engine, see pipeline.py) ships no Tamil
voice at all -- only this module's separate engine can speak Tamil today.
It's kept in its own module and only imported when actually needed (see
pipeline.synthesize_speech()'s local import) -- English/Hindi users never
pay this cost.

Gated behind CONFIG.tamil_tts_enabled (default True); the caller is
responsible for checking that flag before importing this module.

Nothing manual is required to actually use this: if coqui-tts isn't
installed yet, _load_model() below installs requirements-indic-tts.txt
itself (once, on first real Tamil synthesis) and retries the import,
rather than making a human run pip by hand first. Same for the model
checkpoint (~1.5GB: FastPitch acoustic model + HiFiGAN vocoder) -- not
committed to the repo, downloaded from AI4Bharat's GitHub release on
first use and cached under models/ai4bharat_tts/. Both of those first-use
costs are also kicked off proactively in the background at server
startup (see chat_web.py's predownload_in_background()) so they're
usually already done by the time a real request needs them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
import zipfile

_synthesizer = None

_REQUIREMENTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "requirements-indic-tts.txt"
)
_CHECKPOINT_URL = (
    "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/ta.zip"
)
_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "ai4bharat_tts"
)
_CHECKPOINT_DIR = os.path.join(_MODELS_DIR, "ta")
_FASTPITCH_DIR = os.path.join(_CHECKPOINT_DIR, "fastpitch")
_HIFIGAN_DIR = os.path.join(_CHECKPOINT_DIR, "hifigan")


def _ensure_dependencies_installed() -> None:
    """Imports coqui-tts's Synthesizer; on failure, installs
    requirements-indic-tts.txt with the current interpreter's pip and
    retries once. Raises the retry's own ImportError (unmodified) if
    imports still fail afterwards, so a genuine problem (no internet, no
    compiler for a package that needs one, etc.) surfaces as a normal
    traceback rather than silently retrying forever.
    """
    try:
        from TTS.utils.synthesizer import Synthesizer  # noqa: F401

        return
    except ImportError:
        pass

    print(
        "[indic_tts] Tamil TTS dependencies not found -- installing "
        f"{_REQUIREMENTS_FILE} now (one-time, ~1GB of packages)..."
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", _REQUIREMENTS_FILE], check=True
    )
    print("[indic_tts] Tamil TTS dependencies installed.")
    import importlib

    importlib.invalidate_caches()


def _checkpoint_present() -> bool:
    return os.path.isfile(os.path.join(_FASTPITCH_DIR, "best_model.pth")) and os.path.isfile(
        os.path.join(_HIFIGAN_DIR, "best_model.pth")
    )


def _patch_speakers_file_path() -> None:
    """AI4Bharat's shipped fastpitch/config.json bakes in the absolute
    path from their own training machine for the speaker-embedding file
    (e.g. "models/v1/ta/fastpitch/speakers.pth") -- meaningless on any
    other machine. Worse, the code path that loads this multi-speaker
    FastPitch model (TTS.tts.models.forward_tts.ForwardTTS.init_from_config
    -> SpeakerManager.init_from_config) resolves a relative path against
    the CURRENT PROCESS's cwd, not the config file's own directory, so it
    silently fails wherever chat_web.py happens to be launched from
    (verified live: it resolved to `<cwd>/models/v1/ta/fastpitch/
    speakers.pth`, which doesn't exist anywhere). Rewriting both places
    this path appears (top-level `speakers_file` and the nested
    `model_args.speakers_file`) to this machine's real, absolute path
    once, right after extracting, is the fix -- idempotent, so re-running
    on an already-patched config is harmless.
    """
    config_path = os.path.join(_FASTPITCH_DIR, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    correct_path = os.path.join(_FASTPITCH_DIR, "speakers.pth").replace("\\", "/")
    config["speakers_file"] = correct_path
    config.setdefault("model_args", {})["speakers_file"] = correct_path
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def ensure_checkpoint_downloaded() -> None:
    """Downloads and extracts the ~1.5GB Tamil FastPitch+HiFiGAN
    checkpoint from AI4Bharat's GitHub release if it isn't already
    present under models/ai4bharat_tts/. Safe to call repeatedly (no-op
    once downloaded) and safe to call from a background thread at server
    startup (see chat_web.py) as well as lazily from _load_model() below.
    """
    if _checkpoint_present():
        return

    print(
        f"[indic_tts] Downloading Tamil TTS checkpoint (~1.5GB, one-time) "
        f"from {_CHECKPOINT_URL} ..."
    )
    os.makedirs(_MODELS_DIR, exist_ok=True)
    zip_path = os.path.join(_MODELS_DIR, "ta.zip")
    urllib.request.urlretrieve(_CHECKPOINT_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(_MODELS_DIR)
    os.remove(zip_path)
    print("[indic_tts] Tamil TTS checkpoint downloaded and ready.")


def _load_model():
    global _synthesizer
    if _synthesizer is not None:
        return _synthesizer

    _ensure_dependencies_installed()
    ensure_checkpoint_downloaded()
    _patch_speakers_file_path()  # idempotent -- always re-check/fix, not just right after a fresh
    # download, so a checkpoint placed by hand (or moved/copied from elsewhere) gets the same
    # self-healing treatment rather than silently failing to load (see its own docstring).

    from TTS.utils.synthesizer import Synthesizer

    speakers_file = os.path.join(_FASTPITCH_DIR, "speakers.pth")
    _synthesizer = Synthesizer(
        tts_checkpoint=os.path.join(_FASTPITCH_DIR, "best_model.pth"),
        tts_config_path=os.path.join(_FASTPITCH_DIR, "config.json"),
        tts_speakers_file=speakers_file if os.path.exists(speakers_file) else None,
        tts_languages_file=None,
        vocoder_checkpoint=os.path.join(_HIFIGAN_DIR, "best_model.pth"),
        vocoder_config=os.path.join(_HIFIGAN_DIR, "config.json"),
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=False,  # deliberate -- see module docstring
    )
    return _synthesizer


def synthesize_to_file(text: str, out_path: str) -> str:
    """Synthesizes Tamil `text` to a wav file at `out_path`. Loads the
    model once (module-level cache) and reuses it on subsequent calls.
    """
    synth = _load_model()  # installs deps + downloads checkpoint first if needed

    # This checkpoint is a multi-speaker model (male + female) and needs a
    # speaker_name on every call; which one we pick doesn't matter for our
    # use case (a single consistent voice per language, same as Piper's
    # one voice per language), so just pick whichever name sorts first --
    # deterministic across calls, and works whether the checkpoint ships
    # 1 or 2 speakers.
    kwargs = {}
    speaker_manager = getattr(synth.tts_model, "speaker_manager", None)
    speaker_names = getattr(speaker_manager, "speaker_names", None) if speaker_manager else None
    if speaker_names:
        kwargs["speaker_name"] = sorted(speaker_names)[0]

    wav = synth.tts(text, **kwargs)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    synth.save_wav(wav, out_path)
    return out_path
