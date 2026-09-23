"""
Tests for pipeline.py's in-process Piper voice caching (_load_piper_voice)
and the synthesize_speech() English/Hindi path built on it. Added when
synthesize_speech() moved from shelling out to the `piper` CLI on every
call (reloading the ~60MB ONNX model from disk each time) to a cached
in-process piper.voice.PiperVoice per language, mirroring indic_tts.py's
own _load_model() caching pattern. Mocked at the PiperVoice boundary --
this suite never loads a real voice model, matching conftest.py's "no
Piper binary actually invoked" guarantee.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pipeline


class _FakeVoice:
    def __init__(self):
        self.synthesize_wav_calls: list[tuple[str, wave.Wave_write]] = []

    def synthesize_wav(self, text: str, wav_file: wave.Wave_write) -> None:
        self.synthesize_wav_calls.append((text, wav_file))
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 100)  # 100 frames of silence -- just needs to be valid


class TestLoadPiperVoice:
    def setup_method(self):
        pipeline._piper_voices.clear()

    def teardown_method(self):
        pipeline._piper_voices.clear()

    def test_loads_once_and_caches(self, monkeypatch):
        load_calls = []

        class _FakePiperVoice:
            @staticmethod
            def load(model_path):
                load_calls.append(model_path)
                return _FakeVoice()

        monkeypatch.setattr("piper.voice.PiperVoice", _FakePiperVoice)

        first = pipeline._load_piper_voice("en")
        second = pipeline._load_piper_voice("en")

        assert len(load_calls) == 1
        assert first is second

    def test_separate_cache_per_language(self, monkeypatch):
        load_calls = []

        class _FakePiperVoice:
            @staticmethod
            def load(model_path):
                load_calls.append(model_path)
                return _FakeVoice()

        monkeypatch.setattr("piper.voice.PiperVoice", _FakePiperVoice)

        en_voice = pipeline._load_piper_voice("en")
        hi_voice = pipeline._load_piper_voice("hi")

        assert len(load_calls) == 2
        assert en_voice is not hi_voice

    def test_unknown_language_falls_back_to_english_voice_path(self, monkeypatch):
        requested_paths = []

        class _FakePiperVoice:
            @staticmethod
            def load(model_path):
                requested_paths.append(model_path)
                return _FakeVoice()

        monkeypatch.setattr("piper.voice.PiperVoice", _FakePiperVoice)

        pipeline._load_piper_voice("fr")  # not in PIPER_VOICES

        assert requested_paths == [pipeline.PIPER_VOICES["en"]]


class TestSynthesizeSpeechPiperPath:
    def setup_method(self):
        pipeline._piper_voices.clear()

    def teardown_method(self):
        pipeline._piper_voices.clear()

    def test_writes_a_valid_nonempty_wav(self, tmp_path, monkeypatch):
        fake_voice = _FakeVoice()
        monkeypatch.setattr(pipeline, "_load_piper_voice", lambda language: fake_voice)

        out_path = str(tmp_path / "out.wav")
        result = pipeline.synthesize_speech("Hello there.", out_path=out_path, language="en")

        assert result == out_path
        with wave.open(out_path, "rb") as wav_file:
            assert wav_file.getnframes() > 0
        assert len(fake_voice.synthesize_wav_calls) == 1

    def test_text_is_truncated_before_reaching_the_voice(self, tmp_path, monkeypatch):
        fake_voice = _FakeVoice()
        monkeypatch.setattr(pipeline, "_load_piper_voice", lambda language: fake_voice)

        long_text = "word " * 200  # comfortably over MAX_SPEECH_CHARS, no punctuation
        pipeline.synthesize_speech(long_text, out_path=str(tmp_path / "out.wav"), language="en")

        synthesized_text, _wav_file = fake_voice.synthesize_wav_calls[0]
        # truncate_for_speech falls back to a hard cut + "..." when no
        # sentence boundary exists in range (this text has none), so the
        # result can run a few chars past MAX_SPEECH_CHARS for the ellipsis.
        assert len(synthesized_text) <= pipeline.MAX_SPEECH_CHARS + len("...")
        assert synthesized_text.endswith("...")
        assert len(long_text) > pipeline.MAX_SPEECH_CHARS  # sanity: input really was over the cap
