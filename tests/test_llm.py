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

@pytest.mark.parametrize("headers, expected", [
    ({"retry-after-ms": "1250"}, 1.25),
    ({"retry-after": "3"}, 3.0),
    ({"retry-after-ms": "bad", "retry-after": "4"}, 4.0),
    ({"retry-after-ms": "-5", "retry-after": "2"}, 2.0),
    ({"retry-after-ms": "inf", "retry-after": "nan"}, None),
    ({"retry-after": "garbage"}, None),
])
def test_retry_header_parsing(headers, expected):
    exc = StatusError(429)
    exc.response = SimpleNamespace(headers=headers)
    translated = translate_error(exc)
    assert isinstance(translated, LLMTransientError)
    assert translated.retry_after_s == expected


def test_retry_after_http_date(monkeypatch):
    from reviewharness.llm import groq_provider
    monkeypatch.setattr(groq_provider.time, "time", lambda: 1_700_000_000)
    from email.utils import formatdate
    exc = StatusError(503)
    exc.response = SimpleNamespace(headers={"retry-after": formatdate(1_700_000_005, usegmt=True)})
    assert translate_error(exc).retry_after_s == 5


def test_auth_and_bad_request_never_receive_retry_delay():
    for status in (400, 401):
        exc = StatusError(status)
        exc.response = SimpleNamespace(headers={"retry-after": "2"})
        assert isinstance(translate_error(exc), LLMPermanentError)


def test_header_errors_redact_secrets():
    exc = StatusError(429, "gsk_abcdefghijklmnop1234")
    exc.response = SimpleNamespace(headers={"retry-after": "2"})
    err = translate_error(exc)
    assert err.retry_after_s == 2
    assert "gsk_abcdefghijklmnop1234" not in str(err)


def test_existing_transient_constructor_compatible():
    assert LLMTransientError("temporary").retry_after_s is None


def test_shared_gate_respects_rate_limit_across_calls():
    from reviewharness.llm.groq_provider import _RateGate
    clock = [100.0]
    waits = []

    def advance(delay):
        waits.append(delay)
        clock[0] += delay

    gate = _RateGate(clock=lambda: clock[0], sleep=advance)
    error = StatusError(429)
    error.response = SimpleNamespace(headers={"retry-after": "9"})
    with pytest.raises(StatusError):
        gate.run(lambda: (_ for _ in ()).throw(error))
    assert gate.run(lambda: "ok") == "ok"
    assert waits == [9.0]


def test_shared_gate_rejects_unbounded_cooldown_without_sleeping():
    from reviewharness.llm.groq_provider import _RateGate
    gate = _RateGate(clock=lambda: 100.0, sleep=lambda duration: pytest.fail("must not sleep"))
    error = StatusError(429)
    error.response = SimpleNamespace(headers={"retry-after": "60"})
    with pytest.raises(StatusError):
        gate.run(lambda: (_ for _ in ()).throw(error))
    with pytest.raises(LLMTransientError, match="safe wait bound") as caught:
        gate.run(lambda: pytest.fail("must not call"))
    assert caught.value.retry_after_s == 60


def test_rate_limit_message_redacts_organization():
    from reviewharness.logging_setup import redact
    secret = "org_01kzv9g03eeh58ax4mtwace59w"
    assert secret not in redact("rate limited: " + secret)
