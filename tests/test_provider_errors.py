import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.llm.exceptions import LLMProviderError
from app.llm.providers.cloudflare import CloudflareProvider
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.groq import GroqProvider
from app.llm.providers.llm7 import LLM7Provider
from app.llm.providers.openrouter import OpenRouterProvider


@pytest.mark.parametrize(
    "provider_type",
    [
        GroqProvider,
        OpenRouterProvider,
        GeminiProvider,
        CloudflareProvider,
        LLM7Provider,
    ],
)
def test_provider_uses_sanitized_shared_error_mapping(provider_type) -> None:
    secret = "sensitive-sdk-error"
    endpoint = "https://provider.invalid/private-endpoint"
    create = AsyncMock(
        side_effect=RuntimeError(f"{secret} {endpoint}"),
    )
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
        llm7_enabled=True,
        llm7_api_key="llm7-test-key",
    )
    provider = provider_type(settings=settings, client=client)

    with pytest.raises(LLMProviderError) as captured:
        asyncio.run(
            provider.chat([{"role": "user", "content": "Hello"}]),
        )

    assert type(captured.value) is LLMProviderError
    assert secret not in str(captured.value)
    assert endpoint not in str(captured.value)
    assert captured.value.__cause__ is None
