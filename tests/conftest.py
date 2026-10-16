"""
Shared pytest fixtures. The whole suite runs with no GPU, no Ollama, no
microphone, and no Piper binary actually invoked -- every external
dependency is mocked at its call site (see individual test files), which
is what lets this run in CI on a plain runner.
"""

from __future__ import annotations

import dataclasses
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


@pytest.fixture
def learning_config(tmp_path, monkeypatch):
    """Swaps chat_web.CONFIG for one with the continuous-learning layer
    turned on and pointed at a throwaway SQLite file -- for tests that
    exercise src/learning/ through chat_web.py's endpoints (the `client`
    fixture above, with the real module-level CONFIG, always has
    learning_enabled=False). Only affects the test that asks for it.
    """
    import chat_web
    import config as config_module

    test_config = dataclasses.replace(
        config_module.Config(),
        learning_enabled=True,
        learning_db_path=str(tmp_path / "learning.db"),
    )
    monkeypatch.setattr(chat_web, "CONFIG", test_config)
    return test_config


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 -- any failure just means "no usable GPU"
        return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked `gpu` when no CUDA device is usable, so the
    same suite runs unchanged on CI runners and on a developer GPU box.
    (The shipped suite mocks all GPU work, so today no test carries the
    marker; it exists for future real-hardware tests.)
    """
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
