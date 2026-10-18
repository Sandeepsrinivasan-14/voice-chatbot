"""
Tests for src/learning/correction_scoring.py -- pure scoring primitives,
no I/O, no database. Known input/output pairs only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from learning import correction_scoring as scoring


class TestStringSimilarity:
    def test_identical_strings_score_one(self):
        assert scoring.string_similarity("four", "four") == 1.0

    def test_case_insensitive(self):
        assert scoring.string_similarity("Four", "four") == 1.0

    def test_completely_different_strings_score_low(self):
        assert scoring.string_similarity("four", "xyz") < 0.3


class TestContextSimilarity:
    def test_identical_context_scores_one(self):
        assert scoring.context_similarity("the cat sat", "the cat sat") == 1.0

    def test_disjoint_context_scores_zero(self):
        assert scoring.context_similarity("apple banana", "car train") == 0.0

    def test_empty_context_scores_zero_not_one(self):
        # An absent context must never itself justify a correction.
        assert scoring.context_similarity("", "") == 0.0
        assert scoring.context_similarity("hello", "") == 0.0

    def test_partial_overlap_is_between_zero_and_one(self):
        score = scoring.context_similarity("meet me at four", "call me at four")
        assert 0.0 < score < 1.0


class TestFrequencyScore:
    def test_saturates_at_the_configured_count(self):
        assert scoring.frequency_score(5, saturation=5) == 1.0
        assert scoring.frequency_score(50, saturation=5) == 1.0

    def test_below_saturation_scales_linearly(self):
        assert scoring.frequency_score(1, saturation=5) == pytest.approx(0.2)

    def test_zero_frequency_scores_zero(self):
        assert scoring.frequency_score(0, saturation=5) == 0.0


class TestRecencyScore:
    def test_just_now_scores_near_one(self):
        now = datetime.now(timezone.utc)
        score = scoring.recency_score(now.isoformat(), halflife_days=30, now=now)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_one_halflife_ago_scores_near_half(self):
        now = datetime.now(timezone.utc)
        then = now - timedelta(days=30)
        score = scoring.recency_score(then.isoformat(), halflife_days=30, now=now)
        assert score == pytest.approx(0.5, rel=0.05)

    def test_malformed_timestamp_scores_zero_not_raises(self):
        assert scoring.recency_score("not-a-date", halflife_days=30) == 0.0


class TestSttUncertaintyBonus:
    def test_unknown_confidence_is_neutral(self):
        assert scoring.stt_uncertainty_bonus(None) == 0.0

    def test_confident_stt_gets_no_bonus(self):
        assert scoring.stt_uncertainty_bonus(0.0) == 0.0

    def test_uncertain_stt_gets_a_larger_bonus_than_confident_stt(self):
        assert scoring.stt_uncertainty_bonus(-1.2) > scoring.stt_uncertainty_bonus(-0.1)


class TestBlendedScore:
    def test_strong_match_scores_higher_than_weak_match(self):
        now = datetime.now(timezone.utc)
        strong = scoring.blended_score(
            phrase_similarity=1.0,
            context_sim=1.0,
            frequency=5,
            frequency_saturation=5,
            last_seen_at=now.isoformat(),
            recency_halflife_days=30,
            avg_logprob=-0.2,
            now=now,
        )
        weak = scoring.blended_score(
            phrase_similarity=0.5,
            context_sim=0.0,
            frequency=1,
            frequency_saturation=5,
            last_seen_at=now.isoformat(),
            recency_halflife_days=30,
            avg_logprob=-0.2,
            now=now,
        )
        assert strong > weak
        assert strong <= 1.0
