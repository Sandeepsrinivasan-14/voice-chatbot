"""
Tests for src/learning/correction_engine.py -- the deterministic scoring
orchestration that decides keep / suggest / auto_apply. The central
guarantee under test is that a learned correction can never turn into a
blanket find-and-replace: it only fires where the surrounding context
actually resembles what it was learned from.
"""

from __future__ import annotations

from learning import correction_engine, correction_memory, feedback_store

DB_NAME = "feedback.db"
PROFILE = "alice"

_DEFAULT_KWARGS = dict(
    suggest_threshold=0.6,
    auto_apply_threshold=0.92,
    frequency_saturation=5,
    recency_halflife_days=30,
    context_aware_enabled=True,
    allow_auto_apply=True,
)


def _seed_rule(db_path, sentence_with_original, corrected_word, original_word, repeats=1):
    """Seeds a correction_rules row the realistic way: via a feedback event
    whose diff carries the surrounding context, exactly like
    /api/feedback would produce.
    """
    corrected_sentence = sentence_with_original.replace(original_word, corrected_word)
    for _ in range(repeats):
        event = feedback_store.insert_feedback_event(
            db_path,
            profile_id=PROFILE,
            session_id="s1",
            original_text=sentence_with_original,
            corrected_text=corrected_sentence,
        )
        correction_memory.upsert_rule_from_event(db_path, event)


class TestAntiGlobalReplace:
    """The 'four' -> 'for' example from the design: this must never become
    a blanket find-and-replace across unrelated sentences.
    """

    def test_same_phrase_in_unrelated_context_is_not_suggested(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed_rule(db_path, "meet me at four o'clock", "for", "four", repeats=3)

        suggestions = correction_engine.suggest_corrections(
            db_path, PROFILE, "I have four apples", **_DEFAULT_KWARGS
        )
        assert suggestions == []

    def test_same_phrase_in_matching_context_is_suggested(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed_rule(db_path, "meet me at four o'clock", "for", "four", repeats=3)

        suggestions = correction_engine.suggest_corrections(
            db_path, PROFILE, "let's meet at four o'clock tomorrow", **_DEFAULT_KWARGS
        )
        assert len(suggestions) == 1
        assert suggestions[0].original_phrase == "four"
        assert suggestions[0].corrected_phrase == "for"


class TestConservativeDefaults:
    def test_no_matching_rule_means_no_suggestions(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        suggestions = correction_engine.suggest_corrections(
            db_path, PROFILE, "hello world", **_DEFAULT_KWARGS
        )
        assert suggestions == []

    def test_single_observation_never_auto_applies(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed_rule(db_path, "meet me at four o'clock", "for", "four", repeats=1)

        suggestions = correction_engine.suggest_corrections(
            db_path, PROFILE, "let's meet at four o'clock tomorrow", **_DEFAULT_KWARGS
        )
        assert all(s.band != "auto_apply" for s in suggestions)


class TestBandGating:
    def test_auto_apply_capped_at_suggest_when_context_aware_disabled(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed_rule(db_path, "meet me at four o'clock", "for", "four", repeats=10)

        kwargs = dict(_DEFAULT_KWARGS, context_aware_enabled=False, suggest_threshold=0.3)
        suggestions = correction_engine.suggest_corrections(
            db_path, PROFILE, "meet me at four o'clock", **kwargs
        )
        assert len(suggestions) == 1
        assert suggestions[0].band == "suggest"

    def test_auto_apply_capped_at_suggest_when_flag_disabled(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed_rule(db_path, "meet me at four o'clock", "for", "four", repeats=10)

        kwargs = dict(_DEFAULT_KWARGS, allow_auto_apply=False, auto_apply_threshold=0.85)
        suggestions = correction_engine.suggest_corrections(
            db_path, PROFILE, "meet me at four o'clock", **kwargs
        )
        assert len(suggestions) == 1
        assert suggestions[0].score >= 0.85
        assert suggestions[0].band == "suggest"

    def test_auto_apply_reachable_when_everything_aligned(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed_rule(db_path, "meet me at four o'clock", "for", "four", repeats=10)

        kwargs = dict(_DEFAULT_KWARGS, auto_apply_threshold=0.85)
        suggestions = correction_engine.suggest_corrections(
            db_path, PROFILE, "meet me at four o'clock", **kwargs
        )
        assert len(suggestions) == 1
        assert suggestions[0].band == "auto_apply"


class TestPerProfileIsolation:
    def test_another_profiles_rules_are_never_used(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed_rule(db_path, "meet me at four o'clock", "for", "four", repeats=10)

        suggestions = correction_engine.suggest_corrections(
            db_path, "bob", "meet me at four o'clock", **_DEFAULT_KWARGS
        )
        assert suggestions == []


class TestApplyAutoCorrections:
    def test_applies_only_auto_apply_band_suggestions(self):
        from learning.correction_engine import Suggestion

        text = "meet me at four o'clock"
        suggestions = [
            Suggestion(
                rule_id=1,
                original_phrase="four",
                corrected_phrase="for",
                span=(3, 4),
                score=0.95,
                band="auto_apply",
            ),
            Suggestion(
                rule_id=2,
                original_phrase="o'clock",
                corrected_phrase="oclock",
                span=(4, 5),
                score=0.7,
                band="suggest",
            ),
        ]
        new_text, applied = correction_engine.apply_auto_corrections(text, suggestions)
        assert new_text == "meet me at for o'clock"
        assert len(applied) == 1
        assert applied[0].rule_id == 1

    def test_no_op_when_nothing_in_auto_apply_band(self):
        new_text, applied = correction_engine.apply_auto_corrections("hello world", [])
        assert new_text == "hello world"
        assert applied == []
