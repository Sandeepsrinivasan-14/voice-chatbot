"""
Tests for the Flask chat web front end (src/chat_web.py). Every external
dependency (audio decode, whisper, Ollama, Piper) is mocked -- these
tests exercise endpoint logic (validation, response shape, confidence
gating, streaming proxy behavior), not GPU/network integration. See
tests/conftest.py for the `client` fixture.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
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
            lambda model, audio, language=None: (
                "hello world",
                _fake_conf_result(Decision.ACCEPTED, -0.2),
                "en",
            ),
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
            lambda model, audio, language=None: (
                "garbled",
                _fake_conf_result(Decision.ASK_TO_REPEAT, -0.9),
                "en",
            ),
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

    def test_success_returns_wav_bytes(self, client, monkeypatch):
        import soundfile as sf

        def _fake_synthesize(text, out_path, language=None):
            sf.write(out_path, np.zeros(1600, dtype=np.float32), 16000)
            return out_path

        monkeypatch.setattr(chat_web.pipeline, "synthesize_speech", _fake_synthesize)

        resp = client.post("/api/tts", json={"text": "hello there"})
        assert resp.status_code == 200
        assert resp.mimetype == "audio/wav"
        assert len(resp.data) > 0

    def test_each_request_gets_a_unique_path_and_cleans_up(self, client, monkeypatch):
        # Regression test: /api/tts used to always write to one shared,
        # fixed file (pipeline.synthesize_speech's own default out_path),
        # which risks two concurrent requests (real under waitress, which
        # is multi-threaded) racing on the same path. Every call must now
        # get its own path, and that temp file must not be left behind.
        import soundfile as sf

        seen_paths = []

        def _fake_synthesize(text, out_path, language=None):
            seen_paths.append(out_path)
            sf.write(out_path, np.zeros(1600, dtype=np.float32), 16000)
            return out_path

        monkeypatch.setattr(chat_web.pipeline, "synthesize_speech", _fake_synthesize)

        client.post("/api/tts", json={"text": "first"})
        client.post("/api/tts", json={"text": "second"})

        assert len(seen_paths) == 2
        assert seen_paths[0] != seen_paths[1]
        for path in seen_paths:
            assert not os.path.exists(path)  # cleaned up after being read

    def test_synthesis_failure_is_500(self, client, monkeypatch):
        def _raise(text, out_path, language=None):
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
        monkeypatch.setattr(
            chat_web.requests, "post", lambda *a, **k: _FakeStreamingResponse(lines)
        )

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
        # generate() only reads these CONFIG attributes.
        monkeypatch.setattr(
            chat_web,
            "CONFIG",
            SimpleNamespace(
                ollama_max_retries=0,
                ollama_timeout_s=1,
                ollama_model="llama3.2:3b",
                ollama_model_ta="",
                ollama_auto_pull_ta_model=False,
            ),
        )

        resp = client.post("/api/respond", json={"message": "hi", "history": []})
        assert resp.status_code == 200
        assert "Could not reach Ollama" in resp.get_data(as_text=True)


class TestTranscribeLearningDisabled:
    """CONFIG.learning_enabled defaults to False -- the `client` fixture
    uses the real module-level CONFIG, so /api/transcribe's response here
    must be exactly what it was before the continuous-learning layer
    existed (plus `language`, added independently for Hindi/Tamil
    support): no *learning*-related keys, nothing to accidentally break
    an existing caller.
    """

    def test_response_key_set_is_unchanged(self, client, monkeypatch):
        monkeypatch.setattr(
            chat_web, "decode_browser_audio", lambda raw, sr: np.zeros(16000, dtype=np.float32)
        )
        monkeypatch.setattr(
            chat_web.pipeline,
            "transcribe_with_confidence",
            lambda model, audio, language=None: (
                "hello world",
                _fake_conf_result(Decision.ACCEPTED, -0.2),
                "en",
            ),
        )
        data = {"audio": (io.BytesIO(b"fake webm bytes"), "clip.webm")}
        resp = client.post("/api/transcribe", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert set(resp.get_json().keys()) == {
            "text",
            "decision",
            "avg_logprob",
            "threshold",
            "accepted",
            "whisper_device",
            "language",
        }


class TestTranscribeLearningEnabled:
    def test_profile_id_defaults_to_local_and_no_suggestions_without_history(
        self, client, learning_config, monkeypatch
    ):
        monkeypatch.setattr(
            chat_web, "decode_browser_audio", lambda raw, sr: np.zeros(16000, dtype=np.float32)
        )
        monkeypatch.setattr(
            chat_web.pipeline,
            "transcribe_with_confidence",
            lambda model, audio, language=None: (
                "hello world",
                _fake_conf_result(Decision.ACCEPTED, -0.2),
                "en",
            ),
        )
        data = {"audio": (io.BytesIO(b"fake webm bytes"), "clip.webm")}
        resp = client.post("/api/transcribe", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["profile_id"] == "local"
        assert body["text"] == "hello world"  # untouched -- no learned corrections exist yet
        assert "correction_suggestions" not in body
        assert "auto_corrected" not in body
        # event_id is what makes the browser UI's "Correct" button appear
        # at all (see web_chat/index.html) -- it must actually be sent.
        assert body["event_id"]

    def test_a_bug_in_the_learning_layer_never_breaks_transcription(
        self, client, learning_config, monkeypatch
    ):
        monkeypatch.setattr(
            chat_web, "decode_browser_audio", lambda raw, sr: np.zeros(16000, dtype=np.float32)
        )
        monkeypatch.setattr(
            chat_web.pipeline,
            "transcribe_with_confidence",
            lambda model, audio, language=None: (
                "hello world",
                _fake_conf_result(Decision.ACCEPTED, -0.2),
                "en",
            ),
        )

        def _boom(*args, **kwargs):
            raise RuntimeError("learning layer exploded")

        monkeypatch.setattr(chat_web.correction_engine, "suggest_corrections", _boom)

        data = {"audio": (io.BytesIO(b"fake webm bytes"), "clip.webm")}
        resp = client.post("/api/transcribe", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["text"] == "hello world"
        assert body["accepted"] is True


class TestFeedbackEndpoint:
    def test_disabled_by_default_returns_400(self, client):
        resp = client.post(
            "/api/feedback",
            json={
                "original_text": "I went to Chennai with John",
                "corrected_text": "I went to Chennai with Jon",
            },
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_missing_texts_is_400(self, client, learning_config):
        resp = client.post("/api/feedback", json={"original_text": "hello"})
        assert resp.status_code == 400

    def test_identical_texts_is_400(self, client, learning_config):
        resp = client.post(
            "/api/feedback",
            json={"original_text": "hello world", "corrected_text": "hello world"},
        )
        assert resp.status_code == 400

    def test_valid_correction_is_stored_with_a_detected_diff(self, client, learning_config):
        resp = client.post(
            "/api/feedback",
            json={
                "original_text": "I went to Chennai with John",
                "corrected_text": "I went to Chennai with Jon",
                "profile_id": "alice",
            },
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True
        assert body["diff"] == [
            {"op": "replace", "original_phrase": "John", "corrected_phrase": "Jon"}
        ]

        from learning import feedback_store

        events = feedback_store.query_feedback_events(
            learning_config.learning_db_path, profile_id="alice"
        )
        assert len(events) == 1
        assert events[0].original_text == "I went to Chennai with John"
        assert events[0].corrected_text == "I went to Chennai with Jon"


class TestAudioPersistence:
    """LEARNING_PERSIST_AUDIO_ENABLED is off by default even when learning
    is on (see the `learning_config` fixture); these tests turn it on
    explicitly and verify audio is written to disk only for a
    transcription that actually gets corrected, never for one that
    doesn't.
    """

    def _enable_audio_persistence(self, learning_config, monkeypatch, tmp_path):
        audio_config = dataclasses.replace(
            learning_config,
            learning_persist_audio_enabled=True,
            learning_audio_dir=str(tmp_path / "audio"),
        )
        monkeypatch.setattr(chat_web, "CONFIG", audio_config)
        return audio_config

    def test_nothing_is_written_until_a_correction_is_submitted(
        self, client, learning_config, monkeypatch, tmp_path
    ):
        self._enable_audio_persistence(learning_config, monkeypatch, tmp_path)
        monkeypatch.setattr(
            chat_web, "decode_browser_audio", lambda raw, sr: np.zeros(16000, dtype=np.float32)
        )
        monkeypatch.setattr(
            chat_web.pipeline,
            "transcribe_with_confidence",
            lambda model, audio, language=None: (
                "hello world",
                _fake_conf_result(Decision.ACCEPTED, -0.2),
                "en",
            ),
        )

        data = {"audio": (io.BytesIO(b"fake webm bytes"), "clip.webm")}
        resp = client.post("/api/transcribe", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200

        audio_dir = tmp_path / "audio"
        assert not audio_dir.exists() or not list(audio_dir.iterdir())

    def test_correcting_a_transcript_persists_its_audio(
        self, client, learning_config, monkeypatch, tmp_path
    ):
        self._enable_audio_persistence(learning_config, monkeypatch, tmp_path)
        monkeypatch.setattr(
            chat_web, "decode_browser_audio", lambda raw, sr: np.zeros(16000, dtype=np.float32)
        )
        monkeypatch.setattr(
            chat_web.pipeline,
            "transcribe_with_confidence",
            lambda model, audio, language=None: (
                "hello world",
                _fake_conf_result(Decision.ACCEPTED, -0.2),
                "en",
            ),
        )

        data = {"audio": (io.BytesIO(b"fake webm bytes"), "clip.webm")}
        transcribe_resp = client.post(
            "/api/transcribe", data=data, content_type="multipart/form-data"
        )
        event_id = transcribe_resp.get_json()["event_id"]

        feedback_resp = client.post(
            "/api/feedback",
            json={
                "event_id": event_id,
                "original_text": "hello world",
                "corrected_text": "hello word",
            },
        )
        assert feedback_resp.status_code == 200

        audio_dir = tmp_path / "audio"
        files = list(audio_dir.iterdir())
        assert len(files) == 1
        assert files[0].name == f"{event_id}.wav"

        from learning import feedback_store

        events = feedback_store.query_feedback_events(
            str(tmp_path / "learning.db"), profile_id="local"
        )
        assert events[-1].audio_ref == os.path.join(str(tmp_path / "audio"), f"{event_id}.wav")

    def test_uncorrected_transcript_audio_is_never_persisted_even_later(
        self, client, learning_config, monkeypatch, tmp_path
    ):
        self._enable_audio_persistence(learning_config, monkeypatch, tmp_path)
        monkeypatch.setattr(
            chat_web, "decode_browser_audio", lambda raw, sr: np.zeros(16000, dtype=np.float32)
        )
        monkeypatch.setattr(
            chat_web.pipeline,
            "transcribe_with_confidence",
            lambda model, audio, language=None: (
                "hello world",
                _fake_conf_result(Decision.ACCEPTED, -0.2),
                "en",
            ),
        )
        data = {"audio": (io.BytesIO(b"fake webm bytes"), "clip.webm")}
        client.post("/api/transcribe", data=data, content_type="multipart/form-data")
        # No /api/feedback call -- this transcription is never corrected.

        audio_dir = tmp_path / "audio"
        assert not audio_dir.exists() or not list(audio_dir.iterdir())
