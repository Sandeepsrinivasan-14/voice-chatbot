"""
Shared helper: decode whatever a browser's MediaRecorder produced
(webm/opus in Chrome/Edge, ogg/opus in Firefox) into float32 mono PCM,
using PyAV -- already a faster-whisper dependency, so no extra decode
library is needed just for this. Used by both src/record_web.py (Phase 4
browser recorder) and src/chat_web.py (the live voice chat web UI).
"""

from __future__ import annotations

import io

import numpy as np


def decode_browser_audio(raw_bytes: bytes, sample_rate: int) -> np.ndarray:
    """Decode a browser-recorded audio blob to float32 mono PCM at
    `sample_rate`. PyAV bundles its own ffmpeg libs, so this needs no
    system-level ffmpeg install.
    """
    import av

    container = av.open(io.BytesIO(raw_bytes))
    try:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        # flush anything buffered in the resampler
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1))
    finally:
        container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)
