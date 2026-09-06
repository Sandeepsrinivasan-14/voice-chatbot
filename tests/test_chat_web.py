"""
Tests for the Flask chat web front end (src/chat_web.py). Every external
dependency (audio decode, whisper, Ollama, Piper) is mocked -- these
tests exercise endpoint logic (validation, response shape, confidence
gating, streaming proxy behavior), not GPU/network integration. See
tests/conftest.py for the `client` fixture.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np

import chat_web
from confidence_check import ConfidenceResult, Decision


def _fake_conf_result(decision, avg_logprob=-0.3):
    return ConfidenceResult(
        text="hello",
        avg_logprob=avg_logprob,
        decision=decision,
        threshold=-0.6,
        beam_size_used=1,
        attempt=1,
    )


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True


class TestTranscribe:
    def test_missing_audio_file_is_400(self, client):
        resp = client.post("/api/transcribe")
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_accepted_transcription(self, client, monkeypatch):
        monkeypatch.setattr(
            chat_web, "decode_browser_audio", lambda raw, sr: np.zeros(16000, dtype=np.float32)
        )
        monkeypatch.setattr(
            chat_web.pipeline,
            "transcribe_with_confidence",
            lambda model, audio: ("hello world", _fake_conf_result(Decision.ACCEPTED, -0.2)),
        )
        data = {"audio": (io.BytesIO(b"fake webm bytes"), "clip.webm")}
        resp = client.post("/api/transcribe", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["text"] == "hello world"
        assert body["accepted"] is True
        assert body["decision"] == "accepted"

    def test_rejected_transcription_reports_not_accepted(self, client, monkeypatch):
        monkeypatch.setattr(
            chat_web, "decode_browser_audio", lambda raw, sr: np.zeros(16000, dtype=np.float32)
        )
        monkeypatch.setattr(
            chat_web.pipeline,
            "transcribe_with_confidence",
            lambda model, audio: ("garbled", _fake_conf_result(Decision.ASK_TO_REPEAT, -0.9)),
        )
        data = {"audio": (io.BytesIO(b"fake webm bytes"), "clip.webm")}
        resp = client.post("/api/transcribe", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["accepted"] is False
        assert body["decision"] == "ask_to_repeat"

    def test_decode_failure_is_400(self, client, monkeypatch):
        def _raise(raw, sr):
            raise ValueError("bad codec")

        monkeypatch.setattr(chat_web, "decode_browser_audio", _raise)
        data = {"audio": (io.BytesIO(b"not audio"), "clip.webm")}
        resp = client.post("/api/transcribe", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_too_short_recording_is_400(self, client, monkeypatch):
        monkeypatch.setattr(
            chat_web, "decode_browser_audio", lambda raw, sr: np.zeros(100, dtype=np.float32)
        )
        data = {"audio": (io.BytesIO(b"tiny"), "clip.webm")}
        resp = client.post("/api/transcribe", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400


class TestTts:
    def test_empty_text_is_400(self, client):
        resp = client.post("/api/tts", json={"text": ""})
        assert resp.status_code == 400

    def test_success_returns_wav_bytes(self, client, monkeypatch, tmp_path):
        import soundfile as sf

        wav_path = tmp_path / "out.wav"
        sf.write(str(wav_path), np.zeros(1600, dtype=np.float32), 16000)
        monkeypatch.setattr(chat_web.pipeline, "synthesize_speech", lambda text: str(wav_path))

        resp = client.post("/api/tts", json={"text": "hello there"})
        assert resp.status_code == 200
        assert resp.mimetype == "audio/wav"
        assert len(resp.data) > 0

    def test_synthesis_failure_is_500(self, client, monkeypatch):
        def _raise(text):
            raise RuntimeError("piper exploded")

        monkeypatch.setattr(chat_web.pipeline, "synthesize_speech", _raise)
        resp = client.post("/api/tts", json={"text": "hello"})
        assert resp.status_code == 500


class _FakeStreamingResponse:
    """Minimal stand-in for requests.Response as used by respond_endpoint:
    a context manager whose iter_lines() yields Ollama-shaped NDJSON.
    """

    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class TestRespond:
    def test_empty_message_is_400(self, client):
        resp = client.post("/api/respond", json={"message": "", "history": []})
        assert resp.status_code == 400

    def test_streams_ollama_tokens(self, client, monkeypatch):
        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}).encode(),
            json.dumps({"message": {"content": " there"}, "done": False}).encode(),
            json.dumps({"message": {"content": ""}, "done": True}).encode(),
        ]
        monkeypatch.setattr(chat_web.requests, "post", lambda *a, **k: _FakeStreamingResponse(lines))

        resp = client.post("/api/respond", json={"message": "hi", "history": []})
        assert resp.status_code == 200
        assert resp.get_data(as_text=True) == "Hello there"

    def test_history_is_forwarded_to_ollama(self, client, monkeypatch):
        captured = {}
        done_line = json.dumps({"message": {"content": ""}, "done": True}).encode()

        def _capture_post(url, json=None, **kwargs):
            captured["messages"] = json["messages"]
            return _FakeStreamingResponse([done_line])

        monkeypatch.setattr(chat_web.requests, "post", _capture_post)
        history = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        client.post("/api/respond", json={"message": "follow up", "history": history})

        sent_messages = captured["messages"]
        assert sent_messages[0]["role"] == "system"  # SYSTEM_PROMPT always first
        assert {"role": "user", "content": "earlier question"} in sent_messages
        assert sent_messages[-1] == {"role": "user", "content": "follow up"}

    def test_connection_error_surfaces_as_text_without_retry_delay(self, client, monkeypatch):
        import requests as real_requests

        def _always_fail(*a, **k):
            raise real_requests.exceptions.ConnectionError("no server")

        monkeypatch.setattr(chat_web.requests, "post", _always_fail)
        # Zero retries so the test doesn't sleep; respond_endpoint's
        # generate() only reads these two CONFIG attributes.
        monkeypatch.setattr(
            chat_web, "CONFIG", SimpleNamespace(ollama_max_retries=0, ollama_timeout_s=1)
        )

        resp = client.post("/api/respond", json={"message": "hi", "history": []})
        assert resp.status_code == 200
        assert "Could not reach Ollama" in resp.get_data(as_text=True)
