"""
Shared pytest fixtures. The whole suite runs with no GPU, no Ollama, no
microphone, and no Piper binary actually invoked -- every external
dependency is mocked at its call site (see individual test files), which
is what lets this run in CI on a plain runner.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture
def synthetic_audio():
    """A short, quiet synthetic tone -- stands in for a real recording in
    tests that only need *some* valid float32 mono PCM at 16kHz, not real
    speech content.
    """
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    return (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.fixture
def client(monkeypatch):
    """Flask test client for chat_web.app, with the whisper model getter
    replaced so nothing tries to load faster-whisper for real.
    """
    import chat_web

    class _FakeWhisperModel:
        pass

    monkeypatch.setattr(chat_web, "get_whisper_model", lambda: _FakeWhisperModel())
    chat_web.app.config["TESTING"] = True
    with chat_web.app.test_client() as test_client:
        yield test_client
