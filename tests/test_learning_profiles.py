"""
Tests for src/learning/profiles.py -- the local, no-auth profile/session
id defaulting logic. This is the single place that decides what counts as
a valid id, so every other module (and chat_web.py) goes through it rather
than re-deriving the same rule.
"""

from __future__ import annotations

from learning import profiles


class TestResolveProfileId:
    def test_none_becomes_default(self):
        assert profiles.resolve_profile_id(None) == profiles.DEFAULT_PROFILE_ID

    def test_empty_or_whitespace_becomes_default(self):
        assert profiles.resolve_profile_id("") == profiles.DEFAULT_PROFILE_ID
        assert profiles.resolve_profile_id("   ") == profiles.DEFAULT_PROFILE_ID

    def test_valid_id_is_trimmed_and_passed_through(self):
        assert profiles.resolve_profile_id("  alice-1  ") == "alice-1"

    def test_oversized_id_is_truncated_not_rejected(self):
        raw = "a" * 500
        result = profiles.resolve_profile_id(raw)
        assert len(result) == profiles._MAX_ID_LENGTH
        assert result == "a" * profiles._MAX_ID_LENGTH


class TestResolveSessionId:
    def test_none_becomes_default(self):
        assert profiles.resolve_session_id(None) == profiles.DEFAULT_SESSION_ID

    def test_valid_id_passed_through(self):
        assert profiles.resolve_session_id("session-42") == "session-42"
