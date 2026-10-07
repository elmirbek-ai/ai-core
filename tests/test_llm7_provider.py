import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import ValidationError

from app.core.config import Settings
from app.llm.base import BaseLLMProvider
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.providers.llm7 import LLM7Provider


MESSAGES = [{"role": "user", "content": "private prompt"}]
SECRET = "llm7-test-secret"
BASE_URL = "https://llm7.example.invalid/v1"
GENERAL_MODEL = "GLM-5.3-Flash"
REASONING_MODEL = "gpt-oss:20b"
CODE_MODEL = "gpt-oss:20b"


def make_settings(**overrides) -> Settings:
    values = {
        "groq_api_key": "groq-test-key",
        "llm7_enabled": True,
        "llm7_api_key": SECRET,
        "llm7_base_url": f" {BASE_URL} ",
        "llm7_general_model": GENERAL_MODEL,
        "llm7_reasoning_model": REASONING_MODEL,
        "llm7_code_model": CODE_MODEL,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_client(*, result=None, error=None) -> SimpleNamespace:
    create = AsyncMock(return_value=result, side_effect=error)
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
        close=AsyncMock(),
    )


def completion(model: str, content: str | None = "LLM7 reply"):
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content)),
        ],
    )


def status_error(status_code: int, detail: str) -> APIStatusError:
    request = httpx2.Request("POST", f"{BASE_URL}/chat/completions")
    response = httpx2.Response(status_code, request=request)
    return APIStatusError(
        detail,
        response=response,
        body={
            "message": detail,
            "balance": "private-balance",
            "subscription": "private-subscription",
        },
    )


@pytest.mark.parametrize(
    "selected_model",
    [GENERAL_MODEL, REASONING_MODEL, CODE_MODEL],
)
def test_selected_task_model_is_sent(selected_model: str) -> None:
    client = make_client(result=completion(selected_model))
    provider = LLM7Provider(settings=make_settings(), client=client)

    result = asyncio.run(provider.chat(MESSAGES, model=selected_model))

    assert isinstance(provider, BaseLLMProvider)
    assert result == {
        "provider": "llm7",
        "model": selected_model,
        "content": "LLM7 reply",
    }
    client.chat.completions.create.assert_awaited_once_with(
        model=selected_model,
        messages=MESSAGES,
    )


def test_upstream_response_model_is_preserved() -> None:
    client = make_client(result=completion("canonical-upstream-model"))
    provider = LLM7Provider(settings=make_settings(), client=client)

    result = asyncio.run(provider.chat(MESSAGES, model=GENERAL_MODEL))

    assert result["model"] == "canonical-upstream-model"


def test_empty_content_is_safe() -> None:
    client = make_client(result=completion(GENERAL_MODEL, content=None))
    provider = LLM7Provider(settings=make_settings(), client=client)

    result = asyncio.run(provider.chat(MESSAGES))

    assert result["content"] == ""


def test_client_configuration_and_bearer_authentication() -> None:
    settings = make_settings(
        llm7_timeout_seconds=17.0,
        llm7_max_retries=1,
    )

    with patch("app.llm.providers.llm7.AsyncOpenAI") as client_type:
        LLM7Provider(settings=settings)

    client_type.assert_called_once_with(
        api_key=SECRET,
        base_url=BASE_URL,
        timeout=17.0,
        max_retries=1,
    )

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=SECRET, base_url=BASE_URL)
    try:
        assert client.auth_headers == {"Authorization": f"Bearer {SECRET}"}
    finally:
        asyncio.run(client.close())


def test_close_closes_injected_client() -> None:
    client = make_client()
    provider = LLM7Provider(settings=make_settings(), client=client)

    asyncio.run(provider.close())

    client.close.assert_awaited_once()


@pytest.mark.parametrize(
    ("enabled", "api_key"),
    [(False, SECRET), (True, None), (True, "  ")],
)
def test_disabled_or_missing_key_rejects_construction(
    enabled: bool,
    api_key: str | None,
) -> None:
    with pytest.raises(ValueError, match="disabled"):
        LLM7Provider(
            settings=make_settings(
                llm7_enabled=enabled,
                llm7_api_key=api_key,
            )
        )


def test_llm7_config_defaults() -> None:
    settings = Settings(_env_file=None, groq_api_key="groq-test-key")

    assert settings.llm7_enabled is False
    assert settings.llm7_api_key is None
    assert settings.llm7_base_url == "https://api.llm7.io/v1"
    assert settings.llm7_general_model == GENERAL_MODEL
    assert settings.llm7_reasoning_model == REASONING_MODEL
    assert settings.llm7_code_model == CODE_MODEL
    assert settings.llm7_timeout_seconds == 60.0
    assert settings.llm7_max_retries == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm7_base_url", "  "),
        ("llm7_general_model", "  "),
        ("llm7_reasoning_model", "  "),
        ("llm7_code_model", "  "),
        ("llm7_timeout_seconds", 0),
        ("llm7_max_retries", -1),
    ],
)
def test_invalid_llm7_config_is_rejected(field: str, value) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field: value})


@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (status_error(401, "private auth detail"), LLMAuthenticationError),
        (status_error(403, "private permission detail"), LLMAuthenticationError),
        (status_error(429, "private rate detail"), LLMRateLimitError),
        (status_error(400, "private request detail"), LLMProviderError),
        (status_error(500, "private server detail"), LLMUpstreamError),
        (
            APITimeoutError(
                request=httpx2.Request("POST", BASE_URL),
            ),
            LLMTimeoutError,
        ),
        (
            APIConnectionError(
                message="private connection detail",
                request=httpx2.Request("POST", BASE_URL),
            ),
            LLMUpstreamError,
        ),
        (RuntimeError("private unknown detail"), LLMProviderError),
    ],
)
def test_provider_maps_errors_without_leakage(
    error: Exception,
    expected_type: type[LLMProviderError],
) -> None:
    client = make_client(error=error)
    provider = LLM7Provider(settings=make_settings(), client=client)

    with pytest.raises(expected_type) as captured:
        asyncio.run(provider.chat(MESSAGES))

    output = str(captured.value)
    assert "private" not in output
    assert BASE_URL not in output
    assert captured.value.__cause__ is None


@pytest.mark.parametrize(
    "detail",
    ["insufficient_balance", "pro_access_required"],
)
def test_402_is_sanitized_provider_error_without_retry(detail: str) -> None:
    client = make_client(error=status_error(402, detail))
    provider = LLM7Provider(settings=make_settings(), client=client)

    with pytest.raises(LLMProviderError) as captured:
        asyncio.run(provider.chat(MESSAGES))

    assert type(captured.value) is LLMProviderError
    assert detail not in str(captured.value)
    assert "balance" not in str(captured.value)
    assert "subscription" not in str(captured.value)
    client.chat.completions.create.assert_awaited_once()
