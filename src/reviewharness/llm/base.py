"""Provider-neutral contract. Specialists depend on this, never on a vendor SDK."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMError(Exception):
    """Base class. A plain LLMError is treated as permanent (never retried)."""


class LLMTransientError(LLMError):
    """Worth retrying: rate limit, 5xx, connection problem."""


class LLMTimeoutError(LLMTransientError):
    """The provider call exceeded its timeout."""


class LLMPermanentError(LLMError):
    """Retrying cannot help: bad credentials, rejected request."""


class LLMResponseError(LLMError):
    """The call worked but the reply is unusable (empty, cut off). Retried with feedback."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int = 0  # 0 means "not reported"
    completion_tokens: int = 0


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str) -> LLMResult:
        """Return the model's reply text. Raise an LLMError subclass on failure."""
        ...
