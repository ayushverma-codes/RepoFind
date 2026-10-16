import logging

import pytest
from pydantic import ValidationError

from reviewharness.config import ConfigError, Settings
from reviewharness.logging_setup import redact, setup_logging


def test_defaults_use_groq_gpt_oss_20b():
    s = Settings.from_env({})
    assert s.llm_provider == "groq"
    assert s.llm_model == "openai/gpt-oss-20b"
    assert s.max_concurrency == 3
    assert s.test_mode is False


def test_env_overrides_and_blank_ignored():
    s = Settings.from_env({"REVIEW_MAX_RETRIES": "4", "REVIEW_TEST_MODE": "1", "REVIEW_LLM_MODEL": "  "})
    assert s.max_retries == 4
    assert s.test_mode is True
    assert s.llm_model == "openai/gpt-oss-20b"


@pytest.mark.parametrize(
    "env",
    [
        {"REVIEW_MAX_CONCURRENCY": "0"},
        {"REVIEW_MAX_CONCURRENCY": "99"},
        {"REVIEW_MAX_RETRIES": "-1"},
        {"REVIEW_LLM_TIMEOUT_S": "abc"},
        {"REVIEW_LOG_LEVEL": "LOUD"},
    ],
)
def test_invalid_env_rejected(env):
    with pytest.raises(ValidationError):
        Settings.from_env(env)


def test_credentials_required_unless_test_mode():
    with pytest.raises(ConfigError):
        Settings.from_env({}).require_llm_credentials()
    Settings.from_env({"REVIEW_TEST_MODE": "true"}).require_llm_credentials()
    Settings.from_env({"GROQ_API_KEY": "gsk_abcdefghijklmnop"}).require_llm_credentials()


def test_secret_never_leaks():
    key = "gsk_abcdefghijklmnopqrstuv"
    s = Settings.from_env({"GROQ_API_KEY": key})
    assert key not in repr(s)
    assert key not in str(s.public_dict())
    assert s.public_dict()["groq_api_key"] == "set"
    assert Settings.from_env({}).public_dict()["groq_api_key"] == "not set"


def test_redact_masks_keys():
    assert "gsk_abcdefghijklmnop" not in redact("key is gsk_abcdefghijklmnop ok")
    assert "hunter2" not in redact("api_key=hunter2")


def test_logger_redacts_output(capsys):
    logger = setup_logging("INFO")
    logger.info("calling with %s", "gsk_abcdefghijklmnop")
    err = capsys.readouterr().err
    assert "gsk_abcdefghijklmnop" not in err
    assert "[REDACTED]" in err
    assert logging.getLogger("reviewharness").propagate is False
