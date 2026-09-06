"""
Tests for the confidence-aware decision layer -- the circuit breaker that
decides whether a transcription is trustworthy enough to send to the LLM.
"""

from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest

import confidence_check as cc


def _seg(avg_logprob, start=0.0, end=1.0):
    """A minimal stand-in for faster-whisper's Segment -- only the three
    attributes average_logprob()/evaluate() actually read.
    """
    return SimpleNamespace(avg_logprob=avg_logprob, start=start, end=end)


class TestAverageLogprob:
    def test_empty_segments_is_a_hard_reject(self):
        # No speech decoded at all should never accidentally pass the
        # confidence gate -- -999.0 is far below any realistic threshold.
        assert cc.average_logprob([]) == -999.0

    def test_single_segment_returns_its_own_score(self):
        assert cc.average_logprob([_seg(-0.3, 0.0, 1.0)]) == -0.3

    def test_weighted_by_segment_duration(self):
        # A long confident segment should outweigh a short unconfident one.
        segments = [_seg(-0.1, 0.0, 9.0), _seg(-2.0, 9.0, 10.0)]
        score = cc.average_logprob(segments)
        assert -0.3 < score < -0.1  # much closer to the long segment's score

    def test_zero_duration_segment_does_not_divide_by_zero(self):
        segments = [_seg(-0.5, 1.0, 1.0)]  # start == end
        # Should not raise, and should still reflect the segment's score.
        assert cc.average_logprob(segments) == pytest.approx(-0.5)


class TestEvaluate:
    def test_accepts_above_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cc, "LOG_PATH", str(tmp_path / "confidence_log.csv"))
        result = cc.evaluate([_seg(-0.1)], "yes", threshold=-0.6)
        assert result.decision == cc.Decision.ACCEPTED

    def test_first_attempt_below_threshold_retries_with_more_beams(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cc, "LOG_PATH", str(tmp_path / "confidence_log.csv"))
        result = cc.evaluate(
            [_seg(-0.9)], "so", threshold=-0.6, is_retry_attempt=False, attempt=1
        )
        assert result.decision == cc.Decision.RETRY_MORE_BEAMS

    def test_retry_still_below_threshold_asks_to_repeat(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cc, "LOG_PATH", str(tmp_path / "confidence_log.csv"))
        result = cc.evaluate(
            [_seg(-0.9)], "so", threshold=-0.6, is_retry_attempt=True, attempt=2
        )
        assert result.decision == cc.Decision.ASK_TO_REPEAT

    def test_exactly_at_threshold_is_accepted(self, tmp_path, monkeypatch):
        # `>=`, not `>` -- a score exactly on the boundary should pass.
        monkeypatch.setattr(cc, "LOG_PATH", str(tmp_path / "confidence_log.csv"))
        result = cc.evaluate([_seg(-0.6)], "stop", threshold=-0.6)
        assert result.decision == cc.Decision.ACCEPTED


class TestLogging:
    def test_log_decision_writes_a_csv_row(self, tmp_path, monkeypatch):
        log_path = tmp_path / "confidence_log.csv"
        monkeypatch.setattr(cc, "LOG_PATH", str(log_path))

        cc.evaluate([_seg(-0.2)], "help", threshold=-0.6)

        assert log_path.exists()
        with open(log_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["text"] == "help"
        assert rows[0]["decision"] == "accepted"

    def test_multiple_decisions_append_rather_than_overwrite(self, tmp_path, monkeypatch):
        log_path = tmp_path / "confidence_log.csv"
        monkeypatch.setattr(cc, "LOG_PATH", str(log_path))

        cc.evaluate([_seg(-0.2)], "one", threshold=-0.6)
        cc.evaluate([_seg(-0.2)], "two", threshold=-0.6)

        with open(log_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [r["text"] for r in rows] == ["one", "two"]
