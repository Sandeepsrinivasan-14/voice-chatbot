"""
Tests for src/learning/feedback_store.py -- the immutable feedback-event
log. Original and corrected transcripts must be preserved as two distinct
fields, never merged or overwritten, and once written a row must not be
alterable through any code path -- including a raw SQL statement, which is
what the immutability trigger tests below actually exercise (this module
never exposes an update/delete function at all, so that alone would only
prove the absence of a Python-level API, not that the data is truly safe).
"""

from __future__ import annotations

import sqlite3

from learning import feedback_store

DB_NAME = "feedback.db"


class TestInsertFeedbackEvent:
    def test_preserves_original_and_corrected_separately(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        event = feedback_store.insert_feedback_event(
            db_path,
            profile_id="alice",
            session_id="s1",
            original_text="I went to Chennai with John",
            corrected_text="I went to Chennai with Jon",
        )
        assert event.original_text == "I went to Chennai with John"
        assert event.corrected_text == "I went to Chennai with Jon"

        events = feedback_store.query_feedback_events(db_path, profile_id="alice")
        assert len(events) == 1
        assert events[0].original_text == "I went to Chennai with John"
        assert events[0].corrected_text == "I went to Chennai with Jon"

    def test_detects_the_meaningful_diff_at_write_time(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        event = feedback_store.insert_feedback_event(
            db_path,
            profile_id="alice",
            session_id="s1",
            original_text="I went to Chennai with John",
            corrected_text="I went to Chennai with Jon",
        )
        assert len(event.diffs) == 1
        assert event.diffs[0].original_phrase == "John"
        assert event.diffs[0].corrected_phrase == "Jon"

    def test_rejects_empty_original_text(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        try:
            feedback_store.insert_feedback_event(
                db_path,
                profile_id="alice",
                session_id="s1",
                original_text="",
                corrected_text="hello",
            )
        except ValueError as exc:
            assert "non-empty" in str(exc)
        else:
            raise AssertionError("expected ValueError for empty original_text")

    def test_optional_fields_round_trip(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        feedback_store.insert_feedback_event(
            db_path,
            profile_id="alice",
            session_id="s1",
            original_text="four apples",
            corrected_text="for apples",
            context_before="he said",
            stt_confidence=-0.42,
            stt_model_size="small",
            stt_device="cuda",
            language="en",
        )
        stored = feedback_store.query_feedback_events(db_path, profile_id="alice")[0]
        assert stored.context_before == "he said"
        assert stored.stt_confidence == -0.42
        assert stored.stt_model_size == "small"
        assert stored.stt_device == "cuda"
        assert stored.audio_ref is None  # never persisted in this phase


class TestImmutability:
    def test_direct_update_is_rejected_by_the_database(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        feedback_store.insert_feedback_event(
            db_path,
            profile_id="alice",
            session_id="s1",
            original_text="hello world",
            corrected_text="hello word",
        )
        conn = sqlite3.connect(db_path)
        try:
            try:
                conn.execute("UPDATE feedback_events SET original_text = 'tampered'")
            except sqlite3.Error as exc:
                assert "immutable" in str(exc)
            else:
                raise AssertionError("expected the immutability trigger to reject this UPDATE")
        finally:
            conn.close()

    def test_direct_delete_is_rejected_by_the_database(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        feedback_store.insert_feedback_event(
            db_path,
            profile_id="alice",
            session_id="s1",
            original_text="hello world",
            corrected_text="hello word",
        )
        conn = sqlite3.connect(db_path)
        try:
            try:
                conn.execute("DELETE FROM feedback_events")
            except sqlite3.Error as exc:
                assert "immutable" in str(exc)
            else:
                raise AssertionError("expected the immutability trigger to reject this DELETE")
        finally:
            conn.close()

    def test_module_exposes_no_update_or_delete_function(self):
        assert not hasattr(feedback_store, "update_feedback_event")
        assert not hasattr(feedback_store, "delete_feedback_event")


class TestPerProfileIsolation:
    def test_query_scoped_to_profile_excludes_other_profiles(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        feedback_store.insert_feedback_event(
            db_path,
            profile_id="alice",
            session_id="s1",
            original_text="four",
            corrected_text="for",
        )
        feedback_store.insert_feedback_event(
            db_path,
            profile_id="bob",
            session_id="s2",
            original_text="four",
            corrected_text="floor",
        )
        alice_events = feedback_store.query_feedback_events(db_path, profile_id="alice")
        bob_events = feedback_store.query_feedback_events(db_path, profile_id="bob")
        assert len(alice_events) == 1 and alice_events[0].profile_id == "alice"
        assert len(bob_events) == 1 and bob_events[0].profile_id == "bob"

    def test_unscoped_query_returns_all_profiles(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        feedback_store.insert_feedback_event(
            db_path,
            profile_id="alice",
            session_id="s1",
            original_text="four",
            corrected_text="for",
        )
        feedback_store.insert_feedback_event(
            db_path,
            profile_id="bob",
            session_id="s2",
            original_text="four",
            corrected_text="floor",
        )
        all_events = feedback_store.query_feedback_events(db_path)
        assert len(all_events) == 2
