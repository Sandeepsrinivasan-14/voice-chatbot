"""
Pure, deterministic scoring primitives for correction_engine.py.

Every function here takes plain values in and returns a float in [0, 1]
out -- no I/O, no SQLite, no CONFIG access, no model calls of any kind.
Kept separate from correction_engine.py specifically so the scoring math is
unit-testable with plain input/output pairs, independent of the database or
Flask. This is deliberately the "practical factors" list the continuous-
learning design calls for (exact/fuzzy phrase similarity, context
similarity, frequency, recency, STT confidence) and nothing more --
no embeddings, no vector search, no LLM judging.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from difflib import SequenceMatcher

# Relative weight of each scoring component in the final blended score.
# Named constants rather than CONFIG fields -- like diffing.CONTEXT_WINDOW_
# WORDS, these are an implementation detail of the formula, not something
# an end user is expected to retune (the thresholds the formula is compared
# against ARE CONFIG fields -- see config.py's learning_suggest_threshold /
# learning_auto_apply_threshold). Sums to 1.0 so a component in [0,1]
# keeps the blended score in [0,1].
WEIGHT_STRING_SIMILARITY = 0.30
WEIGHT_CONTEXT_SIMILARITY = 0.35
WEIGHT_FREQUENCY = 0.15
WEIGHT_RECENCY = 0.10
WEIGHT_STT_UNCERTAINTY = 0.10

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# Range over which avg_logprob is mapped to an uncertainty bonus. Below
# UNCERTAIN_LOGPROB the bonus saturates at 1.0 (STT was guessing, lean more
# on the learned correction); at/above CONFIDENT_LOGPROB it's 0.0 (STT was
# already sure, don't lean on memory to override it).
_UNCERTAIN_LOGPROB = -1.5
_CONFIDENT_LOGPROB = 0.0


def string_similarity(a: str, b: str) -> float:
    """Exact match -> 1.0; completely different -> close to 0.0. Ratcliff/
    Obershelp similarity via stdlib difflib, not embeddings.
    """
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def context_similarity(context_a: str, context_b: str) -> float:
    """Word-set Jaccard similarity. Two empty contexts are treated as
    having no evidence of matching (0.0, not 1.0) -- an absent context
    should never itself justify a correction.
    """
    words_a = set(_WORD_RE.findall((context_a or "").lower()))
    words_b = set(_WORD_RE.findall((context_b or "").lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def frequency_score(frequency: int, saturation: int) -> float:
    """Diminishing returns past `saturation` occurrences -- a rule seen
    once is weak evidence, a rule seen `saturation`+ times is as strong as
    frequency alone can make it.
    """
    if saturation <= 0:
        return 1.0 if frequency > 0 else 0.0
    return max(0.0, min(1.0, frequency / saturation))


def recency_score(
    last_seen_at: str, halflife_days: float, now: datetime | None = None
) -> float:
    """Exponential decay: a rule last confirmed `halflife_days` ago scores
    0.5, one confirmed just now scores ~1.0, one confirmed long ago decays
    toward 0. Malformed timestamps score 0.0 (treated as unknown/stale)
    rather than raising, since this feeds a "should we trust this" signal,
    not a hard requirement.
    """
    try:
        last_seen = datetime.fromisoformat(last_seen_at)
    except (TypeError, ValueError):
        return 0.0
    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    days_elapsed = max(0.0, (now - last_seen).total_seconds() / 86400.0)
    if halflife_days <= 0:
        return 0.0 if days_elapsed > 0 else 1.0
    return 0.5 ** (days_elapsed / halflife_days)


def stt_uncertainty_bonus(avg_logprob: float | None) -> float:
    """Shakier ASR output (lower avg_logprob) leans more on a learned
    correction; a confident STT result leans less on memory overriding it.
    None (unknown confidence) is neutral: no bonus, no penalty.
    """
    if avg_logprob is None:
        return 0.0
    span = _UNCERTAIN_LOGPROB - _CONFIDENT_LOGPROB  # negative (e.g. -1.5 - 0.0)
    clamped = max(_UNCERTAIN_LOGPROB, min(_CONFIDENT_LOGPROB, avg_logprob))
    return (clamped - _CONFIDENT_LOGPROB) / span


def blended_score(
    *,
    phrase_similarity: float,
    context_sim: float,
    frequency: int,
    frequency_saturation: int,
    last_seen_at: str,
    recency_halflife_days: float,
    avg_logprob: float | None,
    now: datetime | None = None,
) -> float:
    """Combines the components above into one [0, 1] score using the
    module-level weights. This is the single function correction_engine.py
    calls; individual components stay separately unit-testable above.
    """
    return (
        WEIGHT_STRING_SIMILARITY * phrase_similarity
        + WEIGHT_CONTEXT_SIMILARITY * context_sim
        + WEIGHT_FREQUENCY * frequency_score(frequency, frequency_saturation)
        + WEIGHT_RECENCY * recency_score(last_seen_at, recency_halflife_days, now)
        + WEIGHT_STT_UNCERTAINTY * stt_uncertainty_bonus(avg_logprob)
    )
