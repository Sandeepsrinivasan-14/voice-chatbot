"""
Tests for src/learning/correction_memory.py -- derived, rebuildable
per-profile correction rules. The core guarantee this module makes is that
correction_rules can always be safely deleted and rebuilt from
feedback_events without losing anything, and that one profile's rules can
never leak into another's.
"""

from __future__ import annotations

from learning import correction_memory, feedback_store

DB_NAME = "feedback.db"


def _seed(db_path, profile_id, original_text, corrected_text, session_id="s1"):
    event = feedback_store.insert_feedback_event(
        db_path,
        profile_id=profile_id,
        session_id=session_id,
        original_text=original_text,
        corrected_text=corrected_text,
    )
    correction_memory.upsert_rule_from_event(db_path, event)
    return event


class TestUpsertRuleFromEvent:
    def test_first_correction_creates_a_rule(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed(db_path, "alice", "I said four", "I said for")
        rules = correction_memory.get_active_rules(db_path, "alice")
        assert len(rules) == 1
        assert rules[0].original_phrase == "four"
        assert rules[0].corrected_phrase == "for"
        assert rules[0].frequency == 1

    def test_repeated_correction_increments_frequency_not_row_count(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        for _ in range(3):
            _seed(db_path, "alice", "I said four", "I said for")
        rules = correction_memory.get_active_rules(db_path, "alice")
        assert len(rules) == 1
        assert rules[0].frequency == 3

    def test_case_only_change_is_not_turned_into_a_rule(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed(db_path, "alice", "i went to chennai", "i went to Chennai")
        assert correction_memory.get_active_rules(db_path, "alice") == []


class TestRebuild:
    def test_rebuild_does_not_alter_feedback_history(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed(db_path, "alice", "I said four", "I said for")

        before = feedback_store.query_feedback_events(db_path, profile_id="alice")
        correction_memory.rebuild_rules_for_profile(db_path, "alice")
        after = feedback_store.query_feedback_events(db_path, profile_id="alice")

        assert len(before) == len(after) == 1
        assert before[0].original_text == after[0].original_text
        assert before[0].corrected_text == after[0].corrected_text
        assert before[0].event_uuid == after[0].event_uuid

    def test_rebuild_reconstructs_the_same_rule(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed(db_path, "alice", "I said four", "I said for")

        count = correction_memory.rebuild_rules_for_profile(db_path, "alice")
        rules = correction_memory.get_active_rules(db_path, "alice")
        assert count == 1
        assert len(rules) == 1
        assert rules[0].original_phrase == "four"
        assert rules[0].corrected_phrase == "for"

    def test_rebuild_for_one_profile_never_touches_another(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed(db_path, "alice", "I said four", "I said for")
        _seed(db_path, "bob", "I said four", "I said floor")

        correction_memory.rebuild_rules_for_profile(db_path, "alice")

        bob_rules = correction_memory.get_active_rules(db_path, "bob")
        assert len(bob_rules) == 1
        assert bob_rules[0].corrected_phrase == "floor"

    def test_rebuild_all_covers_every_profile_with_history(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed(db_path, "alice", "I said four", "I said for")
        _seed(db_path, "bob", "I said four", "I said floor")

        counts = correction_memory.rebuild_all_rules(db_path)
        assert counts == {"alice": 1, "bob": 1}


class TestDisableRule:
    def test_disabling_excludes_from_active_rules_but_keeps_the_row(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed(db_path, "alice", "I said four", "I said for")
        rule_id = correction_memory.get_active_rules(db_path, "alice")[0].id

        correction_memory.set_rule_enabled(db_path, rule_id, False)
        assert correction_memory.get_active_rules(db_path, "alice") == []

        # Disabling a derived rule must never touch the feedback history
        # it was derived from.
        events = feedback_store.query_feedback_events(db_path, profile_id="alice")
        assert len(events) == 1

    def test_re_enabling_restores_it(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed(db_path, "alice", "I said four", "I said for")
        rule_id = correction_memory.get_active_rules(db_path, "alice")[0].id

        correction_memory.set_rule_enabled(db_path, rule_id, False)
        correction_memory.set_rule_enabled(db_path, rule_id, True)
        assert len(correction_memory.get_active_rules(db_path, "alice")) == 1


class TestPerProfileIsolationInLookup:
    def test_rule_for_one_profile_never_returned_for_another(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed(db_path, "alice", "I said four", "I said for")
        assert correction_memory.get_active_rules(db_path, "bob") == []
