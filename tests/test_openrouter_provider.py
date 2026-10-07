import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import Settings
from app.llm.base import BaseLLMProvider
from app.llm.providers.openrouter import OpenRouterProvider


def openrouter_settings(**overrides) -> Settings:
    values = {
        "groq_api_key": "test-key",
        "openrouter_api_key": "openrouter-test-key",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_openrouter_provider_contract() -> None:
    create = AsyncMock(
        return_value=SimpleNamespace(
            model="openrouter/free",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="fallback response"),
                )
            ],
        )
    )
    close = AsyncMock()
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
        close=close,
    )
    provider = OpenRouterProvider(
        settings=openrouter_settings(),
        client=client,
    )

    async def exercise_provider() -> dict[str, str]:
        result = await provider.chat(
            [{"role": "user", "content": "Hello"}],
        )
        await provider.close()
        return result

    result = asyncio.run(exercise_provider())

    assert isinstance(provider, BaseLLMProvider)
    assert result == {
        "provider": "openrouter",
        "model": "openrouter/free",
        "content": "fallback response",
    }
    close.assert_awaited_once()


def test_openrouter_client_configuration() -> None:
    settings = openrouter_settings(
        openrouter_timeout_seconds=14.0,
        openrouter_max_retries=3,
    )

    with patch("app.llm.providers.openrouter.AsyncOpenAI") as client_type:
        OpenRouterProvider(settings=settings)

    client_type.assert_called_once_with(
        api_key="openrouter-test-key",
        base_url="https://openrouter.ai/api/v1",
        timeout=14.0,
        max_retries=3,
    )


def test_openrouter_provider_rejects_missing_key() -> None:
    settings = openrouter_settings(openrouter_api_key=None)

    with pytest.raises(ValueError, match="provider is disabled"):
        OpenRouterProvider(settings=settings)


def test_openrouter_ignores_primary_groq_model() -> None:
    create = AsyncMock(
        return_value=SimpleNamespace(
            model="openrouter/free",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="fallback response"),
                )
            ],
        )
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
        close=AsyncMock(),
    )
    provider = OpenRouterProvider(
        settings=openrouter_settings(),
        client=client,
    )

    asyncio.run(
        provider.chat(
            [{"role": "user", "content": "Analyze"}],
            model="openai/gpt-oss-120b",
        )
    )

    create.assert_awaited_once_with(
        model="openrouter/free",
        messages=[{"role": "user", "content": "Analyze"}],
    )
