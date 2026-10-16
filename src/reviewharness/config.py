"""Configuration loaded from environment variables.

The API key is held as a SecretStr so it never appears in repr(), logs or
the output of `reviewharness config`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ConfigError(Exception):
    """Raised when configuration is unusable for the requested operation."""


# Environment variable -> Settings field.
_ENV_MAP = {
    "GROQ_API_KEY": "groq_api_key",
    "REVIEW_LLM_MODEL": "llm_model",
    "REVIEW_TEST_MODE": "test_mode",
    "REVIEW_LLM_TIMEOUT_S": "llm_timeout_s",
    "REVIEW_MAX_CONCURRENCY": "max_concurrency",
    "REVIEW_MAX_RETRIES": "max_retries",
    "REVIEW_MAX_REVISIONS": "max_revisions",
    "REVIEW_MAX_REPO_MB": "max_repo_mb",
    "REVIEW_MAX_FILES": "max_files",
    "REVIEW_MAX_FILE_KB": "max_file_kb",
    "REVIEW_TOOL_TIMEOUT_S": "tool_timeout_s",
    "REVIEW_GIT_TIMEOUT_S": "git_timeout_s",
    "REVIEW_TASK_TIMEOUT_S": "task_timeout_s",
    "REVIEW_RUN_TIMEOUT_S": "run_timeout_s",
    "REVIEW_DATA_DIR": "data_dir",
    "REVIEW_LOG_LEVEL": "log_level",
}


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_provider: Literal["groq"] = "groq"
    llm_model: str = Field(default="openai/gpt-oss-20b", min_length=1)
    groq_api_key: Optional[SecretStr] = None
    test_mode: bool = False

    llm_timeout_s: float = Field(default=60.0, gt=0, le=600)
    max_concurrency: int = Field(default=3, ge=1, le=8)
    max_retries: int = Field(default=2, ge=0, le=5)
    max_revisions: int = Field(default=2, ge=0, le=5)
    max_repo_mb: int = Field(default=50, ge=1, le=1024)
    max_files: int = Field(default=10000, ge=1, le=200000)
    max_file_kb: int = Field(default=512, ge=1, le=10240)
    tool_timeout_s: float = Field(default=60.0, gt=0, le=600)
    git_timeout_s: float = Field(default=120.0, gt=0, le=1800)
    task_timeout_s: float = Field(default=300.0, gt=0, le=3600)  # one specialist task, all retries included
    run_timeout_s: float = Field(default=900.0, ge=10, le=7200)

    data_dir: Path = Path(".reviewharness")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        """Build settings from `env` (defaults to os.environ).

        Unset or blank variables fall back to defaults. Invalid values raise
        pydantic.ValidationError.
        """
        source = os.environ if env is None else env
        values: dict[str, Any] = {}
        for var, field in _ENV_MAP.items():
            raw = source.get(var)
            if raw is None or raw.strip() == "":
                continue
            values[field] = raw.strip()
        return cls(**values)

    def require_llm_credentials(self) -> None:
        """Fail clearly if a real provider call would be attempted without a key."""
        if self.test_mode:
            return
        if self.groq_api_key is None or not self.groq_api_key.get_secret_value():
            raise ConfigError(
                "GROQ_API_KEY is not set. Export it, put it in .env, "
                "or set REVIEW_TEST_MODE=1 to run without an API key."
            )

    def public_dict(self) -> dict[str, Any]:
        """Settings safe to print or log (secret replaced by a status string)."""
        data = self.model_dump(mode="json", exclude={"groq_api_key"})
        has_key = bool(self.groq_api_key and self.groq_api_key.get_secret_value())
        data["groq_api_key"] = "set" if has_key else "not set"
        return data
