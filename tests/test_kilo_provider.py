import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.providers.kilo import KiloProvider

MESSAGES = [{"role": "user", "content": "private prompt"}]
BASE_URL = "https://kilo.example.invalid/gateway"
GENERAL_MODEL = "stepfun/step-3.7-flash:free"
CODE_MODEL = "cohere/north-mini-code:free"
LONG_MODEL = "dots-studio/dots-3-note-preview:free"


def make_settings(**overrides) -> Settings:
    values = {
        "groq_api_key": "groq-test-key",
        "kilo_enabled": True,
        "kilo_base_url": f" {BASE_URL}/ ",
        "kilo_general_model": GENERAL_MODEL,
        "kilo_code_model": CODE_MODEL,
        "kilo_long_context_model": LONG_MODEL,
        "kilo_max_retries": 0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def response(status_code: int, body: dict | None = None) -> httpx.Response:
    if body is None:
        body = {
            "error": {
                "message": "private upstream response",
                "metadata": {"user_id": "private-user-id"},
            }
        }
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", f"{BASE_URL}/chat/completions"),
        json=body,
    )


def success_response(
    actual_model: str = "stepfun/step-3.7-flash",
    content: str | None = "Kilo reply",
) -> httpx.Response:
    return response(
        200,
        {
            "model": actual_model,
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"market_cost": 0},
        },
    )


def make_client(*side_effects) -> SimpleNamespace:
    return SimpleNamespace(
        post=AsyncMock(side_effect=list(side_effects)),
        aclose=AsyncMock(),
    )


@pytest.mark.parametrize(
    "selected_model",
    [GENERAL_MODEL, CODE_MODEL, LONG_MODEL],
)
def test_request_uses_selected_task_model_without_authorization(
    selected_model: str,
) -> None:
    client = make_client(success_response(actual_model=selected_model))
    provider = KiloProvider(settings=make_settings(), client=client)

    asyncio.run(provider.chat(MESSAGES, model=selected_model))

    client.post.assert_awaited_once_with(
        f"{BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": selected_model,
            "messages": MESSAGES,
            "stream": False,
        },
    )
    assert "Authorization" not in client.post.await_args.kwargs["headers"]


def test_response_contract_preserves_upstream_model() -> None:
    client = make_client(success_response())
    provider = KiloProvider(settings=make_settings(), client=client)

    result = asyncio.run(provider.chat(MESSAGES, model=GENERAL_MODEL))

    assert result == {
        "provider": "kilo",
        "model": "stepfun/step-3.7-flash",
        "content": "Kilo reply",
    }


def test_empty_content_is_returned_as_empty_string() -> None:
    client = make_client(success_response(content=None))
    provider = KiloProvider(settings=make_settings(), client=client)

    result = asyncio.run(provider.chat(MESSAGES))

    assert result["content"] == ""


def test_close_closes_injected_client() -> None:
    client = make_client()
    provider = KiloProvider(settings=make_settings(), client=client)

    asyncio.run(provider.close())

    client.aclose.assert_awaited_once()


def test_disabled_provider_cannot_be_constructed() -> None:
    with pytest.raises(ValueError, match="disabled"):
        KiloProvider(settings=make_settings(kilo_enabled=False))


def test_kilo_config_defaults_are_opt_in() -> None:
    settings = Settings(_env_file=None, groq_api_key="groq-test-key")

    assert settings.kilo_enabled is False
    assert settings.kilo_base_url == "https://api.kilo.ai/api/gateway"
    assert settings.kilo_general_model == GENERAL_MODEL
    assert settings.kilo_code_model == CODE_MODEL
    assert settings.kilo_long_context_model == LONG_MODEL
    assert settings.kilo_timeout_seconds == 60.0
    assert settings.kilo_max_retries == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kilo_base_url", "  "),
        ("kilo_general_model", "  "),
        ("kilo_code_model", "  "),
        ("kilo_long_context_model", "  "),
        ("kilo_timeout_seconds", 0),
        ("kilo_max_retries", -1),
    ],
)
def test_invalid_kilo_config_is_rejected(field: str, value) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field: value})


@pytest.mark.parametrize(
    ("side_effect", "expected_type"),
    [
        (
            httpx.ReadTimeout(
                "raw timeout",
                request=httpx.Request("POST", BASE_URL),
            ),
            LLMTimeoutError,
        ),
        (
            httpx.ConnectError(
                "raw connection error",
                request=httpx.Request("POST", BASE_URL),
            ),
            LLMUpstreamError,
        ),
        (response(401), LLMAuthenticationError),
        (response(403), LLMAuthenticationError),
        (response(429), LLMRateLimitError),
        (response(400), LLMProviderError),
        (response(500), LLMUpstreamError),
        (RuntimeError("raw unknown error"), LLMProviderError),
    ],
)
def test_provider_maps_native_http_failures(
    side_effect,
    expected_type: type[LLMProviderError],
) -> None:
    client = make_client(side_effect)
    provider = KiloProvider(settings=make_settings(), client=client)

    with pytest.raises(expected_type) as captured:
        asyncio.run(provider.chat(MESSAGES))

    output = str(captured.value)
    assert "private upstream response" not in output
    assert "private-user-id" not in output
    assert BASE_URL not in output


def test_malformed_response_is_sanitized() -> None:
    client = make_client(response(200, {"unexpected": "private payload"}))
    provider = KiloProvider(settings=make_settings(), client=client)

    with pytest.raises(LLMProviderError) as captured:
        asyncio.run(provider.chat(MESSAGES))

    assert "private payload" not in str(captured.value)


@pytest.mark.parametrize(
    "first_failure",
    [
        httpx.ReadTimeout(
            "timeout",
            request=httpx.Request("POST", BASE_URL),
        ),
        response(429),
        response(500),
    ],
)
def test_recoverable_failure_is_retried_then_succeeds(
    monkeypatch,
    first_failure,
) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr("app.llm.providers.kilo.asyncio.sleep", sleep)
    client = make_client(first_failure, success_response())
    provider = KiloProvider(
        settings=make_settings(kilo_max_retries=1),
        client=client,
    )

    result = asyncio.run(provider.chat(MESSAGES))

    assert result["provider"] == "kilo"
    assert client.post.await_count == 2
    sleep.assert_awaited_once_with(0.1)


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_non_recoverable_status_is_not_retried(status_code: int) -> None:
    client = make_client(response(status_code))
    provider = KiloProvider(
        settings=make_settings(kilo_max_retries=1),
        client=client,
    )

    with pytest.raises(LLMProviderError):
        asyncio.run(provider.chat(MESSAGES))

    assert client.post.await_count == 1


def test_max_retries_is_respected(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.providers.kilo.asyncio.sleep", AsyncMock())
    client = make_client(response(429), response(429))
    provider = KiloProvider(
        settings=make_settings(kilo_max_retries=1),
        client=client,
    )

    with pytest.raises(LLMRateLimitError):
        asyncio.run(provider.chat(MESSAGES))

    assert client.post.await_count == 2


def test_sensitive_data_is_not_logged_or_exposed(caplog) -> None:
    sensitive = f"{BASE_URL} private-user-id private prompt"
    client = make_client(RuntimeError(sensitive))
    provider = KiloProvider(settings=make_settings(), client=client)

    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(LLMProviderError) as captured,
    ):
        asyncio.run(provider.chat(MESSAGES))

    output = f"{captured.value} {caplog.text}"
    assert BASE_URL not in output
    assert "private-user-id" not in output
    assert "private prompt" not in output
