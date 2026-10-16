"""LLM provider adapter: one small interface, a Groq implementation, a deterministic fake."""
from reviewharness.llm.base import (
    LLMError,
    LLMPermanentError,
    LLMProvider,
    LLMResponseError,
    LLMResult,
    LLMTimeoutError,
    LLMTransientError,
)
from reviewharness.llm.factory import make_provider

__all__ = [
    "LLMError",
    "LLMPermanentError",
    "LLMProvider",
    "LLMResponseError",
    "LLMResult",
    "LLMTimeoutError",
    "LLMTransientError",
    "make_provider",
]
