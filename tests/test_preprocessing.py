"""
Tests for the audio preprocessing layer.

The most important tests here are the regression tests locking in the
real bug found via data/results/benchmark_results.csv (see
preprocessing.py's "UPDATE, post-benchmark" docstring section): applying
pitch-shift/time-stretch corrections made accuracy dramatically WORSE on
real recordings, so both are disabled by default. These tests make sure
that finding can't silently regress if someone flips the flags back on
without re-benchmarking.
"""

from __future__ import annotations

import numpy as np
import pytest

import preprocessing as pp


class TestNormalizeVolume:
    def test_reaches_target_rms(self):
        audio = (np.random.randn(8000).astype(np.float32)) * 0.5  # loud input
        normalized, _orig_rms, new_rms = pp._normalize_volume(audio)
        assert new_rms == pytest.approx(pp.TARGET_RMS, rel=0.05)

    def test_near_silence_does_not_blow_up(self):
        audio = np.zeros(8000, dtype=np.float32)
        normalized, _orig_rms, _new_rms = pp._normalize_volume(audio)
        assert np.all(np.isfinite(normalized))

    def test_very_quiet_input_does_not_clip_after_gain(self):
        audio = (np.random.randn(8000).astype(np.float32)) * 0.001  # whisper-quiet
        normalized, _orig_rms, _new_rms = pp._normalize_volume(audio)
        assert np.max(np.abs(normalized)) <= 1.0


class TestPitchCorrectionDisabledByDefault:
    """Locks in the benchmark finding: pitch_shift must never be called
    while ENABLE_PITCH_CORRECTION is False, even when detection flags a
    clear outlier -- detection/logging still runs, only the audio
    modification is gated off.
    """

    def test_flag_defaults_to_false(self):
        assert pp.ENABLE_PITCH_CORRECTION is False

    def test_pitch_shift_never_called_even_on_a_flagged_outlier(self, monkeypatch):
        # Force pyin to report a pitch far outside PITCH_MIN_HZ/MAX_HZ --
        # exactly the case that WOULD trigger a correction if the flag
        # were True.
        fake_f0 = np.full(50, 900.0)
        fake_voiced = np.ones(50, dtype=bool)
        monkeypatch.setattr(pp.librosa, "pyin", lambda *a, **k: (fake_f0, fake_voiced, None))

        shift_calls = []
        monkeypatch.setattr(
            pp.librosa.effects,
            "pitch_shift",
            lambda audio, **k: shift_calls.append(k) or audio,
        )

        audio = (np.random.randn(8000).astype(np.float32)) * 0.05
        report = pp.PreprocessingReport()
        out = pp._detect_and_correct_pitch(audio.copy(), 16000, report)

        assert report.pitch_flagged is True  # detection ran and found the outlier
        assert shift_calls == []  # but correction was skipped
        np.testing.assert_array_equal(out, audio)  # audio passed through unchanged


class TestSpeedCorrectionDisabledByDefault:
    """Same guarantee as above, for time_stretch. This is the step that
    caused the dominant accuracy damage (see module docstring) -- onset
    density on a single isolated word is dominated by silence padding,
    not true speaking rate.
    """

    def test_flag_defaults_to_false(self):
        assert pp.ENABLE_SPEED_CORRECTION is False

    def test_time_stretch_never_called_even_on_a_flagged_outlier(self, monkeypatch):
        # A single onset over a long clip -> an extreme "slow" ratio,
        # exactly the case that WOULD trigger a correction if the flag
        # were True.
        monkeypatch.setattr(pp.librosa.onset, "onset_strength", lambda **k: np.zeros(10))
        monkeypatch.setattr(pp.librosa.onset, "onset_detect", lambda **k: np.array([0]))

        stretch_calls = []
        monkeypatch.setattr(
            pp.librosa.effects,
            "time_stretch",
            lambda audio, **k: stretch_calls.append(k) or audio,
        )

        audio = (np.random.randn(16000).astype(np.float32)) * 0.05  # 1s, long enough to estimate
        report = pp.PreprocessingReport()
        out = pp._detect_and_correct_speed(audio.copy(), 16000, report)

        assert report.speed_flagged is True
        assert stretch_calls == []
        np.testing.assert_array_equal(out, audio)


class TestNormalizeAudioEndToEnd:
    """normalize_audio() must never crash the caller -- every internal
    step fails closed (logs + passes audio through) on unexpected input.
    """

    def test_short_clip(self):
        audio = (np.random.randn(1600).astype(np.float32)) * 0.05  # 0.1s
        out, report = pp.normalize_audio(audio, 16000)
        assert np.all(np.isfinite(out))
        assert isinstance(report, pp.PreprocessingReport)

    def test_silence(self):
        audio = np.zeros(16000, dtype=np.float32)
        out, _report = pp.normalize_audio(audio, 16000)
        assert np.all(np.isfinite(out))

    def test_normal_tone(self, synthetic_audio):
        out, report = pp.normalize_audio(synthetic_audio, 16000)
        assert np.all(np.isfinite(out))
        assert report.normalized_rms == pytest.approx(pp.TARGET_RMS, rel=0.1)
