"""
Modulation-robustness layer, part 2: confidence-aware decision layer.

Preprocessing (see preprocessing.py) narrows the acoustic gap, but it
won't close it entirely for every modulation -- an extreme whisper or a
heavily pitch-shifted shout can still come out garbled. The failure mode
that actually hurts a voice assistant isn't "STT sometimes gets it
wrong" -- it's "STT gets it wrong *silently* and the LLM confidently acts
on nonsense." This module is the circuit breaker for that: every
transcription is checked against its own confidence score before it's
allowed downstream.

faster-whisper reports `avg_logprob` per segment -- the average
log-probability the model assigned to the tokens it chose. It's not a
calibrated 0-1 probability, but on this project's own recordings the
same effective bands showed up over and over:
    avg_logprob > -0.5   : model is confident, essentially always correct
    avg_logprob -0.5..-0.9: shaky -- often right, sometimes not
    avg_logprob < -0.9    : model was guessing; frequently wrong

DEFAULT_CONFIDENCE_THRESHOLD starts at -0.6 as a reasonable middle
ground that trades a few unnecessary retries for not letting genuinely
bad transcriptions through. Tune it using logs/confidence_log.csv:
  - Too many false "ask to repeat" on good audio -> lower the threshold
    (more negative, e.g. -0.7) to accept more.
  - Wrong transcriptions reaching the LLM -> raise it (less negative,
    e.g. -0.5) to be stricter.
Because this score distribution shifts per speaker/mic, re-tune it
against your own confidence_log.csv rather than trusting -0.6 blindly.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger("voice_chatbot.confidence_check")

DEFAULT_CONFIDENCE_THRESHOLD = -0.6
LOG_PATH = os.path.join("logs", "confidence_log.csv")
_CSV_HEADER = [
    "timestamp",
    "text",
    "avg_logprob",
    "threshold",
    "decision",
    "beam_size_used",
    "attempt",
]


class Decision(str, Enum):
    ACCEPTED = "accepted"
    RETRY_MORE_BEAMS = "retry_more_beams"
    ASK_TO_REPEAT = "ask_to_repeat"


@dataclass
class ConfidenceResult:
    text: str
    avg_logprob: float
    decision: Decision
    threshold: float
    beam_size_used: int
    attempt: int


def _ensure_log_file() -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_CSV_HEADER)


def log_decision(result: ConfidenceResult) -> None:
    """Append one row per decision to logs/confidence_log.csv for
    auditability -- every accept AND every reject, with the score that
    drove the call, so the threshold can be tuned from evidence later.
    """
    _ensure_log_file()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                result.text,
                f"{result.avg_logprob:.4f}",
                result.threshold,
                result.decision.value,
                result.beam_size_used,
                result.attempt,
            ]
        )


def average_logprob(segments: list) -> float:
    """faster-whisper yields one Segment per detected phrase, each with
    its own avg_logprob. For a short command-word utterance there's
    usually one segment, but average across all of them (weighted by
    length) so a longer sentence isn't unfairly judged by its worst
    segment alone.
    """
    if not segments:
        return -999.0  # no speech decoded at all -> definitely reject
    total_len = sum(max(seg.end - seg.start, 1e-6) for seg in segments)
    weighted = sum(
        seg.avg_logprob * max(seg.end - seg.start, 1e-6) for seg in segments
    )
    return weighted / total_len


def evaluate(
    segments: list,
    text: str,
    *,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    beam_size_used: int = 1,
    attempt: int = 1,
    is_retry_attempt: bool = False,
) -> ConfidenceResult:
    """Decide accept / retry-with-more-beams / ask-to-repeat for one
    transcription attempt, and log the decision.

    Flow used by pipeline.py:
      1. Transcribe with default beam_size (fast path).
      2. evaluate() -- if REJECTED and this was attempt 1, transcribe
         again with a larger beam_size (more thorough search) and
         evaluate() the retry.
      3. If the retry (attempt 2) still fails, decision becomes
         ASK_TO_REPEAT and the caller triggers TTS asking the user to
         repeat themselves rather than guessing.
    """
    score = average_logprob(segments)
    passed = score >= threshold

    if passed:
        decision = Decision.ACCEPTED
    elif not is_retry_attempt:
        decision = Decision.RETRY_MORE_BEAMS
    else:
        decision = Decision.ASK_TO_REPEAT

    result = ConfidenceResult(
        text=text,
        avg_logprob=score,
        decision=decision,
        threshold=threshold,
        beam_size_used=beam_size_used,
        attempt=attempt,
    )
    log_decision(result)
    logger.info(
        "Confidence %.3f (threshold %.2f) -> %s: %r",
        score,
        threshold,
        decision.value,
        text,
    )
    return result
