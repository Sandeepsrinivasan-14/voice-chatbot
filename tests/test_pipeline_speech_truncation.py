"""
Tests for pipeline.truncate_for_speech() -- the hard cap on how much text
actually reaches TTS. Added after a real live-testing failure: a confusing
Tamil input made the LLM ramble for several paragraphs, and indic_tts.py's
old engine (indic-parler-tts, autoregressive and CPU-bound under GPU
memory pressure on a 4GB dev machine) took minutes to synthesize it,
reading to the user as "TTS silently died". SYSTEM_PROMPT already *asks*
for 1-3 sentences but has no way to enforce it -- this does. The cap
stays in place even after switching Tamil to a much faster engine (see
indic_tts.py's docstring): an unbounded reply is still an unbounded wait
no matter how fast the model is per character.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline import MAX_SPEECH_CHARS, MAX_SPEECH_CHARS_TA, truncate_for_speech


class TestTruncateForSpeech:
    def test_short_text_is_untouched(self):
        text = "I am fine, thank you."
        assert truncate_for_speech(text) == text

    def test_exactly_at_limit_is_untouched(self):
        text = "a" * MAX_SPEECH_CHARS
        assert truncate_for_speech(text) == text

    def test_long_text_cuts_at_last_sentence_boundary_within_limit(self):
        first = "This is one short sentence."
        second = "This is another short sentence."
        third = "And a third one that pushes well past the limit here."
        text = " ".join([first, second, third] * 5)  # comfortably over MAX_SPEECH_CHARS
        result = truncate_for_speech(text, max_chars=80)
        assert len(result) <= 80
        assert result.endswith((".", "!", "?"))
        # never mid-sentence: the result must be an exact prefix ending
        # right after a sentence-ending punctuation mark.
        assert text.startswith(result)

    def test_falls_back_to_hard_cut_with_ellipsis_when_no_sentence_boundary(self):
        text = "word " * 200  # one giant run-on with no punctuation at all
        result = truncate_for_speech(text, max_chars=50)
        assert result.endswith("...")
        assert len(result) <= 53

    def test_devanagari_sentence_boundary_is_recognized(self):
        # U+0964 DEVANAGARI DANDA, the Hindi/Sanskrit sentence-end mark.
        text = "र्य् " * 60  # padding well past the limit
        text = "में ठीक हूं। " + text
        result = truncate_for_speech(text, max_chars=30)
        assert result.endswith("।")

    def test_default_cap_is_a_few_sentences_not_a_paragraph(self):
        # Guards against the cap silently regressing to something so large
        # it stops protecting against an unbounded reply / unbounded wait.
        assert 100 <= MAX_SPEECH_CHARS <= 500

    def test_tamil_cap_is_tighter_than_the_default(self):
        # index.html sentence-chunks a reply before ever calling /api/tts,
        # so this cap is a per-CHUNK safety net (one pathological run-on
        # sentence with no punctuation) rather than a whole-reply ratio --
        # it just needs to stay meaningfully under Piper's (English/Hindi)
        # default cap.
        assert MAX_SPEECH_CHARS_TA < MAX_SPEECH_CHARS
        assert MAX_SPEECH_CHARS_TA < 250
