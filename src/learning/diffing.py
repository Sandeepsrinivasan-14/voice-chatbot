"""
Word/phrase-level diff between an original and a corrected transcript.

Treating every correction as one undifferentiated block ("the sentence
changed") throws away exactly the information personalized correction
needs: *which* phrase changed and what it changed to. "I went to Chennai
with John" -> "I went to Chennai with Jon" should detect a single
replace("John", "Jon") change, not "the whole sentence is different now".

Deterministic and dependency-free (stdlib difflib), on purpose -- this is
exactly the kind of thing the project's own confidence gating and
preprocessing modules already do without ML, and a word-level diff needs
none either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

# How many words of surrounding (unchanged) text to capture on each side of
# a change, for later context-aware matching. A module constant, not a
# CONFIG field -- like correction_scoring.py's weights, this is an
# implementation detail of the algorithm rather than a knob end users are
# expected to tune (see config.py's own "not everything is env-overridable"
# precedent).
CONTEXT_WINDOW_WORDS = 4


@dataclass(frozen=True)
class DiffOp:
    op: str  # "replace" | "insert" | "delete"
    original_phrase: str
    corrected_phrase: str
    context_before: str
    context_after: str
    original_index: int  # token index into the original text's tokens
    corrected_index: int  # token index into the corrected text's tokens


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text or "")


def _window(tokens: list[str], start: int, end: int, before: bool) -> str:
    if before:
        lo = max(0, start - CONTEXT_WINDOW_WORDS)
        return " ".join(tokens[lo:start])
    hi = min(len(tokens), end + CONTEXT_WINDOW_WORDS)
    return " ".join(tokens[end:hi])


def word_diff(original_text: str, corrected_text: str) -> list[DiffOp]:
    """Returns the list of meaningful phrase-level changes between the two
    texts. Identical texts (or texts that differ only in whitespace/
    punctuation the tokenizer ignores) return an empty list.
    """
    original_tokens = _tokenize(original_text)
    corrected_tokens = _tokenize(corrected_text)

    matcher = SequenceMatcher(None, original_tokens, corrected_tokens, autojunk=False)
    diffs: list[DiffOp] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        op = "replace" if tag == "replace" else ("insert" if tag == "insert" else "delete")
        diffs.append(
            DiffOp(
                op=op,
                original_phrase=" ".join(original_tokens[i1:i2]),
                corrected_phrase=" ".join(corrected_tokens[j1:j2]),
                context_before=_window(original_tokens, i1, i1, before=True),
                context_after=_window(original_tokens, i2, i2, before=False),
                original_index=i1,
                corrected_index=j1,
            )
        )
    return diffs
