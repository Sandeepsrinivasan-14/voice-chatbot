"""
Tests for src/learning/audio_cache.py -- the short-lived, in-process bridge
between "just transcribed" and "user corrected this a moment later". No
audio is ever written to disk except via persist(), and only for entries
that were actually stashed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import soundfile as sf

from learning import audio_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    # The cache is process-global by design (see module docstring); tests
    # use distinct keys, but clear it anyway so ordering/parallelization
    # can never make one test's leftovers affect another's.
    audio_cache._cache.clear()
    yield
    audio_cache._cache.clear()


def _tone(seconds=0.2, sample_rate=16000):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    return (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class TestStashAndPersist:
    def test_persist_writes_the_stashed_audio(self, tmp_path):
        audio = _tone()
        audio_cache.stash("evt-1", audio, 16000)
        path = audio_cache.persist("evt-1", str(tmp_path), "evt-1")
        assert path is not None
        assert os.path.exists(path)
        written, sr = sf.read(path, dtype="float32")
        assert sr == 16000
        assert len(written) == len(audio)

    def test_persist_without_a_stash_returns_none(self, tmp_path):
        assert audio_cache.persist("never-stashed", str(tmp_path), "x") is None

    def test_persist_consumes_the_entry(self, tmp_path):
        audio_cache.stash("evt-2", _tone(), 16000)
        first = audio_cache.persist("evt-2", str(tmp_path), "evt-2-a")
        second = audio_cache.persist("evt-2", str(tmp_path), "evt-2-b")
        assert first is not None
        assert second is None  # already consumed, not written twice


class TestCacheCap:
    def test_oldest_entries_are_evicted_past_the_cap(self, tmp_path):
        for i in range(audio_cache._MAX_ENTRIES + 5):
            audio_cache.stash(f"evt-{i}", _tone(), 16000)

        # The earliest entries should have been evicted -- never written to
        # disk at all, not even lazily.
        assert audio_cache.persist("evt-0", str(tmp_path), "x") is None
        # A recent one should still be there.
        newest_key = f"evt-{audio_cache._MAX_ENTRIES + 4}"
        assert audio_cache.persist(newest_key, str(tmp_path), "y") is not None
