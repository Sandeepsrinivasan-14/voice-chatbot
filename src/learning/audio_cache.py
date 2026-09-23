"""
Short-lived, in-process cache of recently transcribed audio, keyed by the
correlation id chat_web.py hands back to the browser as `event_id` on
/api/transcribe.

This is deliberately NOT a durable store. Writing every transcription's
audio to disk -- when the overwhelming majority are never corrected --
would be the wrong default for a project that otherwise persists zero
audio. Instead: audio is held in memory, only while
CONFIG.learning_persist_audio_enabled is on, only long enough for the user
to plausibly hit "Correct" on that message, and evicted (never written to
disk) once the cache is full or the process restarts. Permanent storage
only happens via persist(), called from feedback_endpoint() at the moment
a correction is actually submitted for that event_id -- audio for anything
never corrected is never written anywhere.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

import numpy as np
import soundfile as sf

# Small on purpose -- this only needs to bridge "just transcribed" to
# "user clicked Correct a few messages later" for one browser session, not
# serve as real storage.
_MAX_ENTRIES = 50

_lock = threading.Lock()
_cache: "OrderedDict[str, tuple[np.ndarray, int]]" = OrderedDict()


def stash(event_id: str, audio: np.ndarray, sample_rate: int) -> None:
    """Called from /api/transcribe when audio persistence is enabled.
    Overwrites any existing entry for this event_id.
    """
    with _lock:
        _cache[event_id] = (audio, sample_rate)
        _cache.move_to_end(event_id)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)


def persist(event_id: str, audio_dir: str, filename_stem: str) -> str | None:
    """Writes the cached audio for `event_id` to
    <audio_dir>/<filename_stem>.wav and returns that path (relative to the
    project root if audio_dir was given as a relative path), or None if
    nothing was cached for this event_id -- persistence was off when it was
    transcribed, the entry aged out, or the id is unrecognized. The cache
    entry is consumed (removed) either way once looked up.
    """
    with _lock:
        entry = _cache.pop(event_id, None)
    if entry is None:
        return None
    audio, sample_rate = entry
    os.makedirs(audio_dir, exist_ok=True)
    path = os.path.join(audio_dir, f"{filename_stem}.wav")
    sf.write(path, audio, sample_rate)
    return path
