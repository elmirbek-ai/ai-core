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
from app.llm.providers.ollama import OllamaProvider

MESSAGES = [{"role": "user", "content": "Hello"}]
SECRET = "ollama-test-secret"
BASE_URL = "https://ollama.example.invalid"


def make_settings(**overrides) -> Settings:
    values = {
        "groq_api_key": "groq-test-key",
        "ollama_api_key": SECRET,
        "ollama_base_url": f" {BASE_URL}/ ",
        "ollama_model": "gpt-oss:120b",
        "ollama_max_retries": 0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def response(status_code: int, body: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", f"{BASE_URL}/api/chat"),
        json=body or {"error": "private upstream response"},
    )


def success_response() -> httpx.Response:
    return response(
        200,
        {
            "model": "gpt-oss:120b",
            "message": {"role": "assistant", "content": "Ollama reply"},
        },
    )


def make_client(*side_effects) -> SimpleNamespace:
    return SimpleNamespace(
        post=AsyncMock(side_effect=list(side_effects)),
        aclose=AsyncMock(),
    )


def test_request_contract_and_response_contract() -> None:
    client = make_client(success_response())
    provider = OllamaProvider(settings=make_settings(), client=client)

    result = asyncio.run(provider.chat(MESSAGES, model="ignored-model"))

    assert result == {
        "provider": "ollama",
        "model": "gpt-oss:120b",
        "content": "Ollama reply",
    }
    client.post.assert_awaited_once_with(
        f"{BASE_URL}/api/chat",
        headers={
            "Authorization": f"Bearer {SECRET}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-oss:120b",
            "messages": MESSAGES,
            "stream": False,
        },
    )


def test_close_closes_injected_client() -> None:
    client = make_client()
    provider = OllamaProvider(settings=make_settings(), client=client)

    asyncio.run(provider.close())

    client.aclose.assert_awaited_once()


def test_missing_key_disables_provider() -> None:
    with pytest.raises(ValueError, match="API key is missing"):
        OllamaProvider(settings=make_settings(ollama_api_key="  "))


def test_ollama_config_defaults() -> None:
    settings = Settings(_env_file=None, groq_api_key="groq-test-key")

    assert settings.ollama_api_key is None
    assert settings.ollama_base_url == "https://ollama.com"
    assert settings.ollama_model == "gpt-oss:120b"
    assert settings.ollama_timeout_seconds == 60.0
    assert settings.ollama_max_retries == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ollama_base_url", "  "),
        ("ollama_timeout_seconds", 0),
        ("ollama_timeout_seconds", -1),
        ("ollama_max_retries", -1),
    ],
)
def test_invalid_ollama_config_is_rejected(field: str, value) -> None:
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
    provider = OllamaProvider(settings=make_settings(), client=client)

    with pytest.raises(expected_type) as captured:
        asyncio.run(provider.chat(MESSAGES))

    assert SECRET not in str(captured.value)
    assert BASE_URL not in str(captured.value)
    assert "private upstream response" not in str(captured.value)


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
    monkeypatch.setattr("app.llm.providers.ollama.asyncio.sleep", sleep)
    client = make_client(first_failure, success_response())
    provider = OllamaProvider(
        settings=make_settings(ollama_max_retries=2),
        client=client,
    )

    result = asyncio.run(provider.chat(MESSAGES))

    assert result["provider"] == "ollama"
    assert client.post.await_count == 2
    sleep.assert_awaited_once_with(0.1)


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_non_recoverable_status_is_not_retried(status_code: int) -> None:
    client = make_client(response(status_code))
    provider = OllamaProvider(
        settings=make_settings(ollama_max_retries=2),
        client=client,
    )

    with pytest.raises(LLMProviderError):
        asyncio.run(provider.chat(MESSAGES))

    assert client.post.await_count == 1


def test_max_retries_means_initial_attempt_plus_configured_retries(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.llm.providers.ollama.asyncio.sleep",
        AsyncMock(),
    )
    failures = [response(500), response(500), response(500)]
    client = make_client(*failures)
    provider = OllamaProvider(
        settings=make_settings(ollama_max_retries=2),
        client=client,
    )

    with pytest.raises(LLMUpstreamError):
        asyncio.run(provider.chat(MESSAGES))

    assert client.post.await_count == 3


def test_secrets_payload_and_endpoint_are_not_logged(
    caplog,
) -> None:
    client = make_client(RuntimeError(f"{SECRET} {BASE_URL} {MESSAGES}"))
    provider = OllamaProvider(settings=make_settings(), client=client)

    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(LLMProviderError) as captured,
    ):
        asyncio.run(provider.chat(MESSAGES))

    combined_output = f"{captured.value} {caplog.text}"
    assert SECRET not in combined_output
    assert BASE_URL not in combined_output
    assert "Hello" not in combined_output
