from __future__ import annotations

from reviewharness.config import Settings
from reviewharness.llm.base import LLMProvider


def make_provider(settings: Settings) -> LLMProvider:
    """Test mode -> deterministic rule-based fake (no key, no network). Otherwise Groq."""
    if settings.test_mode:
        from reviewharness.llm.fake import RuleBasedProvider

        return RuleBasedProvider()
    from reviewharness.llm.groq_provider import GroqProvider

    return GroqProvider(settings)  # raises ConfigError if the key is missing
