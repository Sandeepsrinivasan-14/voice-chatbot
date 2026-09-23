"""
Word Error Rate evaluation and baseline-vs-candidate comparison for a
fine-tuned STT model.

WER is computed with a plain dynamic-programming edit distance over
whitespace-tokenized, normalized words -- no new dependency (a `jiwer`-
style library would be one more thing to install for something this
small). `transcribe_fn` is injected rather than this module importing
faster-whisper or a HF model directly, so the same evaluation code runs
against either the production CTranslate2 model or a freshly fine-tuned
candidate without caring which.

This module has no opinion on *how* a model produces text -- training.py
handles that. It only answers "is the candidate actually better, on held-
out data, without regressing on individual examples" -- the question the
user's spec requires an answer to before anything gets promoted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

_WORD_RE = re.compile(r"[a-z0-9']+")


def _normalize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard WER: (substitutions + deletions + insertions) / len(reference).
    An empty reference with a non-empty hypothesis scores 1.0 (all
    insertions relative to nothing); two empty strings score 0.0.
    """
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n


@dataclass(frozen=True)
class ExampleResult:
    event_uuid: str
    reference: str
    hypothesis: str
    wer: float


@dataclass(frozen=True)
class EvaluationResult:
    model_label: str
    dataset_version: str
    split: str
    example_count: int
    mean_wer: float
    examples: list[ExampleResult] = field(default_factory=list)


def evaluate_model_on_dataset(
    transcribe_fn: Callable[[str], str],
    manifest_rows: list[dict],
    *,
    model_label: str,
    dataset_version: str,
    split: str = "eval",
) -> EvaluationResult:
    """Runs transcribe_fn(audio_path) -> text over every manifest row in
    `split` (pass split=None to use every row) and scores each against its
    corrected-text reference. Held-out ("eval") rows are what
    compare_models() should be called with -- scoring against the rows a
    candidate model may have effectively "seen" via training would
    overstate how good it is.
    """
    rows = [r for r in manifest_rows if split is None or r.get("split") == split]
    examples = [
        ExampleResult(
            event_uuid=row["event_uuid"],
            reference=row["text"],
            hypothesis=(hyp := transcribe_fn(row["audio_path"])),
            wer=word_error_rate(row["text"], hyp),
        )
        for row in rows
    ]
    mean_wer = sum(e.wer for e in examples) / len(examples) if examples else 0.0
    return EvaluationResult(
        model_label=model_label,
        dataset_version=dataset_version,
        split=split or "all",
        example_count=len(examples),
        mean_wer=mean_wer,
        examples=examples,
    )


@dataclass(frozen=True)
class ComparisonResult:
    baseline_label: str
    candidate_label: str
    baseline_wer: float
    candidate_wer: float
    relative_improvement: float  # positive = candidate is better
    regression_examples: list[str]
    meets_improvement_threshold: bool
    recommendation: str  # "promote" | "reject"


def compare_models(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    *,
    min_relative_improvement: float = 0.05,
    max_per_example_regression: float = 0.5,
) -> ComparisonResult:
    """A candidate is only recommended for promotion if it improves mean
    WER by at least `min_relative_improvement` (relative, not absolute)
    AND does not regress any individual held-out example's WER by more
    than `max_per_example_regression`. Both conditions are required --
    a better average that hides one badly broken example is exactly the
    "unacceptable regression" the user's spec asks to guard against.
    """
    if baseline.mean_wer == 0:
        relative_improvement = 0.0 if candidate.mean_wer == 0 else -1.0
    else:
        relative_improvement = (baseline.mean_wer - candidate.mean_wer) / baseline.mean_wer

    baseline_by_id = {e.event_uuid: e.wer for e in baseline.examples}
    regressions = [
        c.event_uuid
        for c in candidate.examples
        if c.event_uuid in baseline_by_id
        and (c.wer - baseline_by_id[c.event_uuid]) > max_per_example_regression
    ]

    meets = relative_improvement >= min_relative_improvement and not regressions
    return ComparisonResult(
        baseline_label=baseline.model_label,
        candidate_label=candidate.model_label,
        baseline_wer=baseline.mean_wer,
        candidate_wer=candidate.mean_wer,
        relative_improvement=relative_improvement,
        regression_examples=regressions,
        meets_improvement_threshold=meets,
        recommendation="promote" if meets else "reject",
    )
