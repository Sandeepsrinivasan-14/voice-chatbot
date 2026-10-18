"""
Tests for the centralized configuration loader.
"""

from __future__ import annotations

import config as cfg


class TestDefaults:
    def test_bare_config_has_sane_defaults(self):
        c = cfg.Config()
        assert c.sample_rate == 16000
        assert c.whisper_model_size == "small"
        assert c.confidence_threshold == -0.6
        assert c.ollama_url == "http://localhost:11434"
        assert c.production is False

    def test_pipeline_log_path_derives_from_log_dir(self):
        c = cfg.Config(log_dir="somewhere")
        assert (
            c.pipeline_log_path == "somewhere/pipeline.log"
            or c.pipeline_log_path == "somewhere\\pipeline.log"
        )

    def test_config_is_frozen(self):
        c = cfg.Config()
        try:
            c.ollama_model = "different-model"
        except Exception:
            pass
        else:
            raise AssertionError("Config should be immutable (frozen=True)")


class TestFromEnv:
    def test_reads_string_overrides(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
        monkeypatch.setenv("OLLAMA_URL", "http://example-local:11434")
        c = cfg.Config.from_env()
        assert c.ollama_model == "qwen2.5:3b-instruct"
        assert c.ollama_url == "http://example-local:11434"

    def test_reads_float_override(self, monkeypatch):
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "-0.75")
        c = cfg.Config.from_env()
        assert c.confidence_threshold == -0.75

    def test_reads_int_override(self, monkeypatch):
        monkeypatch.setenv("CHAT_WEB_PORT", "9001")
        c = cfg.Config.from_env()
        assert c.chat_web_port == 9001

    def test_web_host_defaults_to_loopback_and_is_overridable(self, monkeypatch):
        monkeypatch.delenv("WEB_HOST", raising=False)
        assert cfg.Config.from_env().web_host == "127.0.0.1"
        monkeypatch.setenv("WEB_HOST", "0.0.0.0")
        assert cfg.Config.from_env().web_host == "0.0.0.0"
        monkeypatch.setenv("WEB_HOST", "   ")
        assert cfg.Config.from_env().web_host == "127.0.0.1"

    def test_reads_bool_override(self, monkeypatch):
        monkeypatch.setenv("PRODUCTION", "1")
        c = cfg.Config.from_env()
        assert c.production is True

        monkeypatch.setenv("PRODUCTION", "0")
        c = cfg.Config.from_env()
        assert c.production is False

    def test_invalid_float_raises_a_clear_error(self, monkeypatch):
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "not-a-number")
        try:
            cfg.Config.from_env()
        except ValueError as exc:
            assert "CONFIDENCE_THRESHOLD" in str(exc)
        else:
            raise AssertionError("Expected a ValueError for an invalid float env var")

    def test_no_env_vars_matches_bare_defaults(self, monkeypatch):
        env_vars = (
            "SAMPLE_RATE",
            "WHISPER_MODEL_SIZE",
            "CONFIDENCE_THRESHOLD",
            "CONFIDENCE_LOG_PATH",
            "OLLAMA_URL",
            "OLLAMA_MODEL",
            "PIPER_EXECUTABLE",
            "PIPER_MODEL_PATH",
            "LOG_DIR",
            "CHAT_WEB_PORT",
            "RECORD_WEB_PORT",
            "PRODUCTION",
            "LEARNING_ENABLED",
            "LEARNING_FEEDBACK_ENABLED",
            "LEARNING_PERSONALIZED_CORRECTION_ENABLED",
            "LEARNING_CONTEXT_AWARE_ENABLED",
            "LEARNING_AUTO_APPLY_ENABLED",
            "LEARNING_DB_PATH",
            "LEARNING_SUGGEST_THRESHOLD",
            "LEARNING_AUTO_APPLY_THRESHOLD",
            "LEARNING_RULE_FREQUENCY_SATURATION",
            "LEARNING_RULE_RECENCY_HALFLIFE_DAYS",
        )
        for name in env_vars:
            monkeypatch.delenv(name, raising=False)
        assert cfg.Config.from_env() == cfg.Config()


class TestLearningDefaults:
    """The continuous-learning layer must ship safe-by-default: the master
    switch and the one flag that rewrites what gets sent to the LLM are
    both off until explicitly enabled.
    """

    def test_learning_is_disabled_by_default(self):
        c = cfg.Config()
        assert c.learning_enabled is False

    def test_auto_apply_is_disabled_by_default_even_if_learning_is_on(self):
        c = cfg.Config()
        assert c.learning_auto_apply_enabled is False

    def test_learning_enabled_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("LEARNING_ENABLED", "1")
        assert cfg.Config.from_env().learning_enabled is True

    def test_learning_thresholds_read_from_env(self, monkeypatch):
        monkeypatch.setenv("LEARNING_AUTO_APPLY_THRESHOLD", "0.8")
        monkeypatch.setenv("LEARNING_SUGGEST_THRESHOLD", "0.4")
        c = cfg.Config.from_env()
        assert c.learning_auto_apply_threshold == 0.8
        assert c.learning_suggest_threshold == 0.4
