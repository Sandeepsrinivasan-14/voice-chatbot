"""
Orchestration: given a transcript and a profile, find candidate learned
corrections, score them, and decide keep / suggest / auto-apply.

This is the only module in src/learning/ that chat_web.py calls directly
at transcribe time (feedback_store/correction_memory are only touched from
the /api/feedback endpoint, when the user actually corrects something).
Every function here is synchronous, in-process SQLite reads plus pure
arithmetic -- no network calls, no Ollama, no embeddings -- so it stays
cheap enough to run on the live transcribe path. chat_web.py still wraps
every call into this module in a try/except; a bug here must never break
transcription.

Anti-global-replace guarantee: a rule matching "four" -> "for" can find
"four" anywhere in a transcript, but unless the *current* surrounding words
overlap with words seen around a past correction of that same phrase for
this same profile, `context_similarity` is 0 and the match is dropped
entirely -- it never reaches "suggest", let alone "auto_apply". See
correction_scoring.py for the scoring math and tests/test_learning_
correction_engine.py for this exact scenario as a regression test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import correction_memory, correction_scoring
from .diffing import CONTEXT_WINDOW_WORDS

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class Suggestion:
    rule_id: int
    original_phrase: str  # the exact span as it appears in this transcript
    corrected_phrase: str
    span: tuple[int, int]  # token index range into this transcript
    score: float
    band: str  # "suggest" | "auto_apply"


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def _find_spans(lowered_tokens: list[str], phrase_tokens: list[str]) -> list[tuple[int, int]]:
    n, m = len(lowered_tokens), len(phrase_tokens)
    if m == 0 or n < m:
        return []
    return [
        (i, i + m) for i in range(n - m + 1) if lowered_tokens[i : i + m] == phrase_tokens
    ]


def _local_context(tokens: list[str], start: int, end: int) -> str:
    lo = max(0, start - CONTEXT_WINDOW_WORDS)
    hi = min(len(tokens), end + CONTEXT_WINDOW_WORDS)
    return " ".join(tokens[lo:start] + tokens[end:hi])


def suggest_corrections(
    db_path: str,
    profile_id: str,
    text: str,
    *,
    context_before: str = "",
    avg_logprob: float | None = None,
    suggest_threshold: float,
    auto_apply_threshold: float,
    frequency_saturation: int,
    recency_halflife_days: float,
    context_aware_enabled: bool,
    allow_auto_apply: bool,
) -> list[Suggestion]:
    """Read-only: looks up this profile's active rules and scores any that
    match a whole-word span in `text`. Never mutates anything. Rules from
    other profiles are structurally unreachable (get_active_rules only
    queries this profile_id).
    """
    rules = correction_memory.get_active_rules(db_path, profile_id)
    if not rules or not text.strip():
        return []

    tokens = _tokenize(text)
    lowered = [t.lower() for t in tokens]
    suggestions: list[Suggestion] = []

    for rule in rules:
        phrase_tokens = rule.original_phrase.split()
        for start, end in _find_spans(lowered, phrase_tokens):
            matched_text = " ".join(tokens[start:end])
            phrase_sim = correction_scoring.string_similarity(
                rule.original_phrase, matched_text.lower()
            )

            if context_aware_enabled:
                current_context = " ".join(
                    [context_before, _local_context(tokens, start, end)]
                )
                learned_context = " ".join(
                    [rule.context_before_sample, rule.context_after_sample]
                )
                context_sim = correction_scoring.context_similarity(
                    learned_context, current_context
                )
                # Hard gate: zero shared context is treated as no evidence
                # at all, not just a low score -- this is what stops a
                # phrase-keyed rule from becoming a global find-and-replace.
                if context_sim <= 0.0:
                    continue
            else:
                context_sim = 0.0  # no context credit when the feature is off

            score = correction_scoring.blended_score(
                phrase_similarity=phrase_sim,
                context_sim=context_sim,
                frequency=rule.frequency,
                frequency_saturation=frequency_saturation,
                last_seen_at=rule.last_seen_at,
                recency_halflife_days=recency_halflife_days,
                avg_logprob=avg_logprob,
            )
            if score < suggest_threshold:
                continue

            band = "suggest"
            # auto_apply additionally requires context-aware scoring to be
            # active -- turning that flag off caps every match at "suggest"
            # even if the raw score would otherwise clear the threshold.
            if context_aware_enabled and allow_auto_apply and score >= auto_apply_threshold:
                band = "auto_apply"

            suggestions.append(
                Suggestion(
                    rule_id=rule.id,
                    original_phrase=matched_text,
                    corrected_phrase=rule.corrected_phrase,
                    span=(start, end),
                    score=round(score, 4),
                    band=band,
                )
            )

    return suggestions


def apply_auto_corrections(
    text: str, suggestions: list[Suggestion]
) -> tuple[str, list[Suggestion]]:
    """Rewrites `text` for every `auto_apply`-band suggestion (first
    matching occurrence only, case-insensitive whole-word). Returns the
    possibly-rewritten text and the subset of suggestions that were
    actually applied (a suggestion is skipped, not force-applied, if its
    exact phrase can no longer be found -- conservative by construction).
    Suggestions not in the `auto_apply` band are left untouched here;
    callers surface them separately for the user to accept manually.
    """
    candidates = [s for s in suggestions if s.band == "auto_apply"]
    if not candidates:
        return text, []

    result = text
    applied: list[Suggestion] = []
    for s in candidates:
        pattern = re.compile(r"\b" + re.escape(s.original_phrase) + r"\b", re.IGNORECASE)
        new_result, n = pattern.subn(s.corrected_phrase, result, count=1)
        if n:
            result = new_result
            applied.append(s)
    return result, applied
