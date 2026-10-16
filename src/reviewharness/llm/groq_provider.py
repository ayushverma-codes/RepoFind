"""Groq chat-completions adapter.

The SDK is imported lazily so the rest of the project (and test mode) works
without it. The SDK's own retries are disabled (max_retries=0): the review graph
owns the retry policy so attempts are bounded and counted in one place.
"""
from __future__ import annotations

from typing import Any, Optional
from email.utils import parsedate_to_datetime
import math
import time

from reviewharness.config import ConfigError, Settings
from reviewharness.llm.base import (
    LLMError,
    LLMPermanentError,
    LLMResponseError,
    LLMResult,
    LLMTimeoutError,
    LLMTransientError,
)
from reviewharness.logging_setup import redact

# gpt-oss models spend completion tokens on reasoning before the JSON answer.
MAX_COMPLETION_TOKENS = 4096


def _retry_delay(exc: Exception) -> float | None:
    """Read public response headers; reject invalid or negative delay values.

    retry-after-ms has precedence when valid; otherwise retry-after accepts
    seconds or HTTP dates. No SDK-private helpers or secrets are used.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    def header(name: str) -> str | None:
        try:
            value = headers.get(name)
            if value is None:
                value = next((v for k, v in headers.items() if str(k).lower() == name), None)
            return value if isinstance(value, str) else None
        except (AttributeError, TypeError, ValueError):
            return None

    def valid(number: float) -> bool:
        return math.isfinite(number) and number >= 0

    ms = header("retry-after-ms")
    if ms is not None:
        try:
            value = float(ms.strip()) / 1000
            if valid(value):
                return value
        except ValueError:
            pass

    seconds = header("retry-after")
    if seconds is not None:
        try:
            value = float(seconds.strip())
            if valid(value):
                return value
        except ValueError:
            pass
        try:
            date = parsedate_to_datetime(seconds.strip())
            if date.tzinfo is None:
                return None
            value = date.timestamp() - time.time()
            if math.isfinite(value):
                return max(0.0, value)
        except (TypeError, ValueError, OverflowError):
            pass
    return None


def translate_error(exc: Exception) -> LLMError:
    """Map an SDK exception to our hierarchy without importing the SDK.

    Matching on class name and `status_code` keeps this testable without groq
    installed. Messages are redacted and truncated before they can be logged.
    """
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    msg = redact(str(exc))[:200]
    if "Timeout" in name:
        return LLMTimeoutError(f"provider timeout: {msg}")
    if isinstance(status, int):
        if status == 429 or status >= 500:
            return LLMTransientError(f"provider returned HTTP {status}: {msg}", retry_after_s=_retry_delay(exc))
        return LLMPermanentError(f"provider rejected the request (HTTP {status}): {msg}")
    if "Connection" in name:
        return LLMTransientError(f"provider connection error: {msg}")
    return LLMError(f"unexpected provider error {name}: {msg}")


class GroqProvider:
    name = "groq"

    def __init__(self, settings: Settings, client: Optional[Any] = None) -> None:
        self.model = settings.llm_model
        if client is None:
            settings.require_llm_credentials()
            try:
                import groq
            except ImportError:
                raise ConfigError(
                    "the 'groq' package is not installed; run: pip install -e \".[agents]\""
                ) from None
            assert settings.groq_api_key is not None
            client = groq.Groq(
                api_key=settings.groq_api_key.get_secret_value(),
                timeout=settings.llm_timeout_s,
                max_retries=0,
            )
        self._client = client

    def complete(self, system: str, user: str) -> LLMResult:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                response_format={"type": "json_object"},
                extra_body={"reasoning_effort": "low"},
            )
        except Exception as exc:  # noqa: BLE001 - translated, never swallowed
            raise translate_error(exc) from None

        try:
            choice = resp.choices[0]
            text = choice.message.content
            finish = getattr(choice, "finish_reason", None)
        except (AttributeError, IndexError, TypeError):
            raise LLMResponseError("provider reply had no choices") from None
        if not text or not text.strip():
            raise LLMResponseError(f"provider returned an empty reply (finish_reason={finish})")

        usage = getattr(resp, "usage", None)
        return LLMResult(
            text=text,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
