"""
Tests for src/learning/diffing.py -- word-level diff detection. The whole
point of this module is to detect *which* phrase changed, not just that
"something" changed, so a correction like "John" -> "Jon" can later be
matched against a specific future occurrence of "John" rather than firing
on the entire corrected sentence.
"""

from __future__ import annotations

from learning import diffing


class TestWordDiff:
    def test_single_word_replace_is_isolated(self):
        # The exact example from the design spec.
        diffs = diffing.word_diff("I went to Chennai with John", "I went to Chennai with Jon")
        assert len(diffs) == 1
        d = diffs[0]
        assert d.op == "replace"
        assert d.original_phrase == "John"
        assert d.corrected_phrase == "Jon"

    def test_identical_text_has_no_diffs(self):
        assert diffing.word_diff("hello there", "hello there") == []

    def test_context_before_captured_from_original(self):
        diffs = diffing.word_diff("I went to Chennai with John", "I went to Chennai with Jon")
        assert "Chennai" in diffs[0].context_before

    def test_pure_insertion_reported_as_insert(self):
        diffs = diffing.word_diff("stop", "please stop")
        assert any(d.op == "insert" and d.corrected_phrase == "please" for d in diffs)

    def test_pure_deletion_reported_as_delete(self):
        diffs = diffing.word_diff("please stop now", "please stop")
        assert any(d.op == "delete" and d.original_phrase == "now" for d in diffs)

    def test_two_independent_single_word_changes_both_detected(self):
        # "cat" -> "bat" and "mat" -> "hat", separated by unchanged text --
        # must not collapse into one "whole sentence changed" diff.
        diffs = diffing.word_diff("cat sat on mat", "bat sat on hat")
        assert len(diffs) == 2
        assert {d.original_phrase for d in diffs} == {"cat", "mat"}
        assert {d.corrected_phrase for d in diffs} == {"bat", "hat"}

    def test_empty_strings_produce_no_diffs(self):
        assert diffing.word_diff("", "") == []
