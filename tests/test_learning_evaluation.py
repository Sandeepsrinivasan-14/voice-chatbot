"""
Tests for src/learning/evaluation.py -- WER computation and baseline-vs-
candidate comparison. No torch/transformers needed here; transcribe_fn is
always a plain injected Python callable, never a real model.
"""

from __future__ import annotations

from learning import evaluation


class TestWordErrorRate:
    def test_identical_text_is_zero(self):
        assert evaluation.word_error_rate("hello world", "hello world") == 0.0

    def test_one_substitution(self):
        assert evaluation.word_error_rate("hello world", "hello there") == 0.5

    def test_case_and_punctuation_insensitive(self):
        assert evaluation.word_error_rate("Hello, world!", "hello world") == 0.0

    def test_empty_reference_and_empty_hypothesis_is_zero(self):
        assert evaluation.word_error_rate("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis_is_one(self):
        assert evaluation.word_error_rate("", "hello") == 1.0

    def test_insertion_increases_wer(self):
        assert evaluation.word_error_rate("hello world", "hello there world") == 0.5


class TestEvaluateModelOnDataset:
    def test_scores_only_the_requested_split(self):
        rows = [
            {"event_uuid": "a", "audio_path": "a.wav", "text": "hello world", "split": "train"},
            {"event_uuid": "b", "audio_path": "b.wav", "text": "hello world", "split": "eval"},
        ]
        calls = []

        def fake_transcribe(path):
            calls.append(path)
            return "hello world"

        result = evaluation.evaluate_model_on_dataset(
            fake_transcribe, rows, model_label="test", dataset_version="v1", split="eval"
        )
        assert calls == ["b.wav"]
        assert result.example_count == 1
        assert result.mean_wer == 0.0

    def test_none_split_scores_everything(self):
        rows = [
            {"event_uuid": "a", "audio_path": "a.wav", "text": "hello", "split": "train"},
            {"event_uuid": "b", "audio_path": "b.wav", "text": "world", "split": "eval"},
        ]
        result = evaluation.evaluate_model_on_dataset(
            lambda p: "wrong", rows, model_label="test", dataset_version="v1", split=None
        )
        assert result.example_count == 2


def _result(label, examples):
    return evaluation.EvaluationResult(
        model_label=label,
        dataset_version="v1",
        split="eval",
        example_count=len(examples),
        mean_wer=sum(e.wer for e in examples) / len(examples),
        examples=examples,
    )


class TestCompareModels:
    def test_recommends_promote_on_clear_improvement(self):
        baseline = _result("base", [
            evaluation.ExampleResult("a", "hello world", "hullo world", 0.5),
            evaluation.ExampleResult("b", "good morning", "good mornin", 0.5),
        ])
        candidate = _result("cand", [
            evaluation.ExampleResult("a", "hello world", "hello world", 0.0),
            evaluation.ExampleResult("b", "good morning", "good morning", 0.0),
        ])
        comparison = evaluation.compare_models(baseline, candidate, min_relative_improvement=0.05)
        assert comparison.recommendation == "promote"
        assert comparison.relative_improvement == 1.0

    def test_rejects_when_improvement_too_small(self):
        baseline = _result("base", [evaluation.ExampleResult("a", "x", "x", 0.10)])
        candidate = _result("cand", [evaluation.ExampleResult("a", "x", "x", 0.099)])
        comparison = evaluation.compare_models(baseline, candidate, min_relative_improvement=0.5)
        assert comparison.recommendation == "reject"

    def test_rejects_on_one_bad_regression_even_if_average_improves(self):
        baseline = _result("base", [
            evaluation.ExampleResult("a", "x", "x", 0.8),
            evaluation.ExampleResult("b", "y", "y", 0.8),
            evaluation.ExampleResult("c", "z", "z", 0.0),
        ])
        candidate = _result("cand", [
            evaluation.ExampleResult("a", "x", "x", 0.0),  # much better
            evaluation.ExampleResult("b", "y", "y", 0.0),  # much better
            evaluation.ExampleResult("c", "z", "z", 0.9),  # badly regressed
        ])
        # Average clearly improves (0.533 -> 0.3), but example "c" regressed
        # by 0.9, well past max_per_example_regression -- must still reject.
        comparison = evaluation.compare_models(
            baseline, candidate, min_relative_improvement=0.05, max_per_example_regression=0.5
        )
        assert comparison.relative_improvement > 0.05
        assert "c" in comparison.regression_examples
        assert comparison.recommendation == "reject"
