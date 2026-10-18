"""
Tests for src/learning/training.py that don't require torch/transformers/
peft to be installed. The module must be importable either way -- nothing
in chat_web.py's import chain ever reaches this file (see its module
docstring) -- and train() must fail with a clear, typed error rather than
a bare ImportError traceback when the training extras aren't present.
"""

from __future__ import annotations

import pytest

from learning import training


class TestImportIsolation:
    def test_module_imports_without_the_training_stack_installed(self):
        # Reaching this line at all proves the import succeeded; asserting
        # on its public surface makes the requirement explicit and named
        # rather than an accident of import order.
        assert hasattr(training, "train")
        assert hasattr(training, "TrainingConfig")


class TestMissingDependencies:
    def test_train_raises_a_clear_error_without_the_training_stack(self, tmp_path):
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("torch is installed in this environment; nothing to assert here")

        with pytest.raises(training.MissingTrainingDependencies):
            training.train(
                str(tmp_path / "manifest.csv"),
                str(tmp_path / "out"),
                config=training.TrainingConfig(),
            )


class TestTrainingConfigDefaults:
    def test_defaults_match_documented_values(self):
        config = training.TrainingConfig()
        assert config.base_model == training.DEFAULT_BASE_MODEL
        assert config.target_modules == training.DEFAULT_LORA_TARGET_MODULES
        assert config.epochs == 3
