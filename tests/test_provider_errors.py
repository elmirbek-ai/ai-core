import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.providers.groq import GroqProvider
from app.llm.providers.cloudflare import CloudflareProvider
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.openrouter import OpenRouterProvider


@pytest.mark.parametrize(
    ("module_name", "provider_type"),
    [
        ("app.llm.providers.groq", GroqProvider),
        ("app.llm.providers.openrouter", OpenRouterProvider),
        ("app.llm.providers.gemini", GeminiProvider),
        ("app.llm.providers.cloudflare", CloudflareProvider),
    ],
)
@pytest.mark.parametrize(
    ("sdk_exception_name", "domain_exception_type"),
    [
        ("AuthenticationError", LLMAuthenticationError),
        ("PermissionDeniedError", LLMAuthenticationError),
        ("RateLimitError", LLMRateLimitError),
        ("APITimeoutError", LLMTimeoutError),
        ("APIConnectionError", LLMUpstreamError),
        ("OpenAIError", LLMProviderError),
    ],
)
def test_sdk_errors_are_mapped_without_internal_detail(
    monkeypatch,
    module_name: str,
    provider_type,
    sdk_exception_name: str,
    domain_exception_type: type[Exception],
) -> None:
    class FakeSDKError(Exception):
        pass

    provider_module = importlib.import_module(module_name)
    monkeypatch.setattr(
        provider_module,
        sdk_exception_name,
        FakeSDKError,
    )
    secret = "sensitive-sdk-error"
    create = AsyncMock(side_effect=FakeSDKError(secret))
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
        close=AsyncMock(),
    )
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openrouter_api_key="openrouter-test-key",
        gemini_api_key="gemini-test-key",
        cloudflare_api_token="cloudflare-test-token",
        cloudflare_account_id="test-account-id",
    )
    provider = provider_type(settings=settings, client=client)

    with pytest.raises(domain_exception_type) as captured:
        asyncio.run(
            provider.chat([{"role": "user", "content": "Hello"}]),
        )

    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None


@pytest.mark.parametrize(
    ("module_name", "provider_type"),
    [
        ("app.llm.providers.groq", GroqProvider),
        ("app.llm.providers.openrouter", OpenRouterProvider),
        ("app.llm.providers.gemini", GeminiProvider),
        ("app.llm.providers.cloudflare", CloudflareProvider),
    ],
)
@pytest.mark.parametrize(
    ("status_code", "domain_exception_type"),
    [
        (500, LLMUpstreamError),
        (503, LLMUpstreamError),
        (400, LLMProviderError),
    ],
)
def test_api_status_errors_use_consistent_domain_semantics(
    monkeypatch,
    module_name: str,
    provider_type,
    status_code: int,
    domain_exception_type: type[Exception],
) -> None:
    class FakeStatusError(Exception):
        def __init__(self, message: str) -> None:
            super().__init__(message)
            self.status_code = status_code

    provider_module = importlib.import_module(module_name)
    monkeypatch.setattr(
        provider_module,
        "APIStatusError",
        FakeStatusError,
    )
    create = AsyncMock(side_effect=FakeStatusError("sensitive-sdk-error"))
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
        close=AsyncMock(),
    )
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openrouter_api_key="openrouter-test-key",
        gemini_api_key="gemini-test-key",
        cloudflare_api_token="cloudflare-test-token",
        cloudflare_account_id="test-account-id",
    )
    provider = provider_type(settings=settings, client=client)

    with pytest.raises(domain_exception_type) as captured:
        asyncio.run(
            provider.chat([{"role": "user", "content": "Hello"}]),
        )

    assert "sensitive-sdk-error" not in str(captured.value)
    assert captured.value.__cause__ is None
