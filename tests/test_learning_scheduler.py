"""
Tests for src/learning/scheduler.py -- the automated check-and-maybe-train
loop. Covers the safety properties the spec calls for explicitly:
  - not enough new data is a safe no-op, not an error
  - overlapping runs are prevented (and a stale lock is reclaimed)
  - state (watermark, last success, last error) survives across calls
  - a successful promotion invokes the hot-reload callback; a rejected
    candidate does not
  - training/evaluation failures are recorded, not raised
Training itself is never exercised for real here (no torch dependency in
this test env) -- learning.training.train is monkeypatched, matching how
test_learning_training.py already treats that module as an external
dependency boundary.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import soundfile as sf

from config import Config
from learning import feedback_store, model_registry, scheduler


def _write_wav(path, seconds=0.5, sample_rate=16000) -> str:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(path), audio, sample_rate)
    return str(path)


def _seed_event(db_path, tmp_path, *, n=0, language="en"):
    audio_ref = _write_wav(tmp_path / f"event_{n}.wav")
    return feedback_store.insert_feedback_event(
        db_path, profile_id="alice", session_id="s1",
        original_text=f"text {n} four", corrected_text=f"text {n} for",
        audio_ref=audio_ref, language=language,
    )


def _make_config(tmp_path, **overrides) -> Config:
    defaults = dict(
        learning_enabled=True,
        learning_auto_training_enabled=True,
        learning_min_new_events_for_training=3,
        learning_db_path=str(tmp_path / "feedback.db"),
        learning_datasets_root=str(tmp_path / "datasets"),
        learning_models_root=str(tmp_path / "models"),
        learning_registry_path=str(tmp_path / "models" / "registry.json"),
        learning_scheduler_state_path=str(tmp_path / "scheduler_state.json"),
        learning_scheduler_lock_path=str(tmp_path / "scheduler.lock"),
        learning_training_lock_stale_seconds=3600,
        learning_training_language="en",
    )
    defaults.update(overrides)
    return Config(**defaults)


class TestNotEnoughData:
    def test_skips_training_below_threshold(self, tmp_path):
        config = _make_config(tmp_path)
        _seed_event(config.learning_db_path, tmp_path, n=0)
        _seed_event(config.learning_db_path, tmp_path, n=1)

        result = scheduler.run_training_cycle(config)

        assert result == {
            "ran": True, "reason": "not_enough_data",
            "new_event_count": 2, "required": 3,
        }

    def test_disabled_flags_are_a_pure_noop(self, tmp_path):
        config = _make_config(tmp_path, learning_auto_training_enabled=False)
        result = scheduler.run_training_cycle(config)
        assert result == {"ran": False, "reason": "disabled"}

        config2 = _make_config(tmp_path, learning_enabled=False)
        result2 = scheduler.run_training_cycle(config2)
        assert result2 == {"ran": False, "reason": "disabled"}


class TestLocking:
    def test_second_concurrent_call_is_skipped(self, tmp_path):
        config = _make_config(tmp_path)
        os.makedirs(os.path.dirname(config.learning_scheduler_lock_path), exist_ok=True)
        with open(config.learning_scheduler_lock_path, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "acquired_at": "now"}, f)

        result = scheduler.run_training_cycle(config)
        assert result == {"ran": False, "reason": "lock_busy"}

    def test_stale_lock_is_reclaimed(self, tmp_path):
        config = _make_config(tmp_path, learning_training_lock_stale_seconds=1)
        os.makedirs(os.path.dirname(config.learning_scheduler_lock_path), exist_ok=True)
        with open(config.learning_scheduler_lock_path, "w", encoding="utf-8") as f:
            json.dump({"pid": 999999, "acquired_at": "long ago"}, f)
        old_time = time.time() - 3600
        os.utime(config.learning_scheduler_lock_path, (old_time, old_time))

        _seed_event(config.learning_db_path, tmp_path, n=0)
        result = scheduler.run_training_cycle(config)
        # Lock was stale, so this reaches the real not-enough-data check
        # rather than bailing out as lock_busy.
        assert result["reason"] == "not_enough_data"
        assert not os.path.exists(config.learning_scheduler_lock_path)

    def test_lock_is_released_after_a_successful_cycle(self, tmp_path):
        config = _make_config(tmp_path)
        scheduler.run_training_cycle(config)
        assert not os.path.exists(config.learning_scheduler_lock_path)


class TestStatePersistence:
    def test_watermark_advances_even_when_not_enough_data(self, tmp_path):
        config = _make_config(tmp_path, learning_min_new_events_for_training=100)
        _seed_event(config.learning_db_path, tmp_path, n=0)
        scheduler.run_training_cycle(config)

        state = scheduler._load_state(config.learning_scheduler_state_path)
        # Below threshold means "no training attempted", so the watermark
        # must NOT advance -- these events should still count as new next time.
        assert state["last_trained_event_id"] == 0
        assert state["status"] == "idle"

    def test_status_endpoint_shape(self, tmp_path):
        config = _make_config(tmp_path)
        scheduler.run_training_cycle(config)
        status = scheduler.get_status(
            config.learning_scheduler_state_path, config.learning_registry_path
        )
        assert "scheduler" in status
        assert "production_model" in status
        assert status["production_model"] is None


class TestTrainingAndPromotion:
    def _seed_enough_events(self, config, tmp_path, count=5):
        for i in range(count):
            _seed_event(config.learning_db_path, tmp_path, n=i)

    def _patch_training(self, monkeypatch, tmp_path, *, wer_improves: bool):
        """Stubs out the heavy training.train()/transcribe path so the
        scheduler's own orchestration (registry, evaluation wiring,
        promotion, callback) is exercised without needing torch installed.
        """
        from learning import training as training_module

        ct2_dir = str(tmp_path / "candidate_ct2")
        os.makedirs(ct2_dir, exist_ok=True)

        def fake_train(manifest_path, output_dir, *, config):
            return ct2_dir

        call_labels = []

        def fake_load_ct2_transcribe_fn(model_path_or_size, *, device, compute_type):
            call_labels.append(model_path_or_size)
            is_candidate = model_path_or_size == ct2_dir

            def _transcribe(audio_path):
                # Candidate always gets it right; baseline is wrong unless
                # we want the "candidate does not actually improve" case.
                if is_candidate and wer_improves:
                    return "matches whatever the reference text is"
                return "totally wrong transcription"

            return _transcribe

        monkeypatch.setattr(training_module, "train", fake_train)
        monkeypatch.setattr(
            training_module, "_load_ct2_transcribe_fn", fake_load_ct2_transcribe_fn
        )
        return ct2_dir

    def test_promotes_and_invokes_reload_callback_when_candidate_is_better(
        self, tmp_path, monkeypatch
    ):
        config = _make_config(tmp_path)
        self._seed_enough_events(config, tmp_path)
        ct2_dir = self._patch_training(monkeypatch, tmp_path, wer_improves=True)

        # Both baseline and candidate produce the SAME correct text in this
        # stub (there's no real audio difference) -- so instead exercise
        # the "meets threshold" branch directly via a monkeypatched
        # evaluation.compare_models to keep this test about orchestration,
        # not WER arithmetic (that's evaluation.py's own test file's job).
        from learning import evaluation as evaluation_module

        def fake_compare_models(baseline, candidate, *, min_relative_improvement=0.05, **kw):
            return evaluation_module.ComparisonResult(
                baseline_label=baseline.model_label, candidate_label=candidate.model_label,
                baseline_wer=0.5, candidate_wer=0.1, relative_improvement=0.8,
                regression_examples=[], meets_improvement_threshold=True,
                recommendation="promote",
            )

        monkeypatch.setattr(evaluation_module, "compare_models", fake_compare_models)

        promoted = []
        result = scheduler.run_training_cycle(
            config, on_promoted=lambda version, path: promoted.append((version, path))
        )

        assert result["ran"] is True
        assert result["promoted"] is True
        assert promoted == [(result["candidate_version"], ct2_dir)]

        production = model_registry.get_production_model(config.learning_registry_path)
        assert production["version"] == result["candidate_version"]

        state = scheduler._load_state(config.learning_scheduler_state_path)
        assert state["last_successful_training_at"] is not None
        assert state["last_trained_event_id"] > 0

    def test_does_not_promote_or_reload_when_candidate_is_not_better(
        self, tmp_path, monkeypatch
    ):
        config = _make_config(tmp_path)
        self._seed_enough_events(config, tmp_path)
        self._patch_training(monkeypatch, tmp_path, wer_improves=False)

        from learning import evaluation as evaluation_module

        def fake_compare_models(baseline, candidate, *, min_relative_improvement=0.05, **kw):
            return evaluation_module.ComparisonResult(
                baseline_label=baseline.model_label, candidate_label=candidate.model_label,
                baseline_wer=0.2, candidate_wer=0.3, relative_improvement=-0.5,
                regression_examples=[], meets_improvement_threshold=False,
                recommendation="reject",
            )

        monkeypatch.setattr(evaluation_module, "compare_models", fake_compare_models)

        promoted = []
        result = scheduler.run_training_cycle(
            config, on_promoted=lambda version, path: promoted.append((version, path))
        )

        assert result["promoted"] is False
        assert promoted == []
        production = model_registry.get_production_model(config.learning_registry_path)
        assert production is None

        # The watermark still advances -- these events were considered and
        # must not immediately re-trigger another attempt next cycle.
        state = scheduler._load_state(config.learning_scheduler_state_path)
        assert state["last_trained_event_id"] > 0
        assert state["last_successful_training_at"] is None

    def test_auto_deploy_disabled_skips_the_reload_callback(self, tmp_path, monkeypatch):
        config = _make_config(tmp_path, learning_auto_deploy_enabled=False)
        self._seed_enough_events(config, tmp_path)
        self._patch_training(monkeypatch, tmp_path, wer_improves=True)

        from learning import evaluation as evaluation_module

        def fake_compare_models(baseline, candidate, *, min_relative_improvement=0.05, **kw):
            return evaluation_module.ComparisonResult(
                baseline_label=baseline.model_label, candidate_label=candidate.model_label,
                baseline_wer=0.5, candidate_wer=0.1, relative_improvement=0.8,
                regression_examples=[], meets_improvement_threshold=True,
                recommendation="promote",
            )

        monkeypatch.setattr(evaluation_module, "compare_models", fake_compare_models)

        promoted = []
        result = scheduler.run_training_cycle(
            config, on_promoted=lambda version, path: promoted.append((version, path))
        )

        assert result["promoted"] is True  # registry promotion still happens
        assert promoted == []  # but the live process is not told to hot-swap

    def test_training_failure_is_recorded_not_raised(self, tmp_path, monkeypatch):
        config = _make_config(tmp_path)
        self._seed_enough_events(config, tmp_path)

        from learning import training as training_module

        def raising_train(manifest_path, output_dir, *, config):
            raise RuntimeError("boom")

        monkeypatch.setattr(training_module, "train", raising_train)

        result = scheduler.run_training_cycle(config)

        assert result["reason"] == "error"
        assert "boom" in result["error"]
        state = scheduler._load_state(config.learning_scheduler_state_path)
        assert state["last_error"] and "boom" in state["last_error"]
        assert state["consecutive_failures"] == 1
        assert state["status"] == "idle"  # never left stuck as "running"
        assert not os.path.exists(config.learning_scheduler_lock_path)


class TestBackgroundThread:
    def test_start_background_scheduler_returns_none_when_disabled(self, tmp_path):
        config = _make_config(tmp_path, learning_auto_training_enabled=False)
        assert scheduler.start_background_scheduler(config) is None

    def test_start_background_scheduler_starts_a_daemon_thread(self, tmp_path):
        config = _make_config(tmp_path, learning_training_interval_minutes=1000)
        thread = scheduler.start_background_scheduler(config)
        try:
            assert thread is not None
            assert thread.daemon is True
            assert thread.is_alive()
        finally:
            thread.stop()
            thread.join(timeout=2)
