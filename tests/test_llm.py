"""LLM adapter: Groq error translation and response handling, tested without the SDK or network."""
from types import SimpleNamespace

import pytest

from reviewharness.config import ConfigError, Settings
from reviewharness.llm import (
    LLMError,
    LLMPermanentError,
    LLMResponseError,
    LLMTimeoutError,
    LLMTransientError,
    make_provider,
)
from reviewharness.llm.fake import RuleBasedProvider, ScriptedProvider
from reviewharness.llm.groq_provider import GroqProvider, translate_error


class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class StatusError(Exception):
    def __init__(self, code, msg="boom"):
        super().__init__(msg)
        self.status_code = code


def client_returning(text, finish="stop", usage=(11, 7), choices=True):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message, finish_reason=finish)
    resp = SimpleNamespace(
        choices=[choice] if choices else [],
        usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]),
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return resp

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def client_raising(exc):
    def create(**kwargs):
        raise exc

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def provider(client):
    return GroqProvider(Settings(), client=client)


def test_groq_success_sends_expected_request():
    client, calls = client_returning('{"findings": []}')
    result = provider(client).complete("SYS", "USER")
    assert result.text == '{"findings": []}'
    assert (result.prompt_tokens, result.completion_tokens) == (11, 7)
    sent = calls[0]
    assert sent["model"] == "openai/gpt-oss-20b"
    assert sent["temperature"] == 0
    assert sent["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


@pytest.mark.parametrize("text", ["", "   ", None])
def test_groq_empty_reply_is_response_error(text):
    client, _ = client_returning(text, finish="length")
    with pytest.raises(LLMResponseError):
        provider(client).complete("s", "u")


def test_groq_no_choices_is_response_error():
    client, _ = client_returning("x", choices=False)
    with pytest.raises(LLMResponseError):
        provider(client).complete("s", "u")


@pytest.mark.parametrize(
    "exc, expected",
    [
        (APITimeoutError("t"), LLMTimeoutError),
        (APIConnectionError("c"), LLMTransientError),
        (StatusError(429), LLMTransientError),
        (StatusError(503), LLMTransientError),
        (StatusError(401), LLMPermanentError),
        (StatusError(400), LLMPermanentError),
    ],
)
def test_groq_error_translation(exc, expected):
    with pytest.raises(expected):
        provider(client_raising(exc)).complete("s", "u")


def test_connection_error_is_not_a_timeout():
    assert not isinstance(translate_error(APIConnectionError("c")), LLMTimeoutError)


def test_unknown_exception_is_plain_llm_error():
    err = translate_error(ValueError("x"))
    assert type(err) is LLMError


def test_error_message_never_contains_api_key():
    err = translate_error(StatusError(401, "bad key gsk_abcdefghijklmnop1234 rejected"))
    assert "gsk_abcdefghijklmnop1234" not in str(err)


def test_make_provider_requires_key_outside_test_mode():
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        make_provider(Settings())


def test_make_provider_test_mode_needs_no_key():
    assert isinstance(make_provider(Settings(test_mode=True)), RuleBasedProvider)


def test_scripted_provider_replays_and_raises():
    p = ScriptedProvider([LLMTransientError("x"), "ok"])
    with pytest.raises(LLMTransientError):
        p.complete("s", "u")
    assert p.complete("s", "u").text == "ok"
    with pytest.raises(RuntimeError):
        p.complete("s", "u")
    assert len(p.calls) == 3
