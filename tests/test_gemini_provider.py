import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import Settings
from app.llm.base import BaseLLMProvider
from app.llm.providers.gemini import GeminiProvider


def gemini_settings(**overrides) -> Settings:
    values = {
        "groq_api_key": "test-key",
        "gemini_api_key": "gemini-test-key",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_gemini_provider_contract() -> None:
    create = AsyncMock(
        return_value=SimpleNamespace(
            model="gemini-3.8-flash",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Gemini response"),
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
    provider = GeminiProvider(
        settings=gemini_settings(),
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
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "content": "Gemini response",
    }
    close.assert_awaited_once()


def test_gemini_client_configuration() -> None:
    settings = gemini_settings(
        gemini_timeout_seconds=16.0,
        gemini_max_retries=3,
    )

    with patch("app.llm.providers.gemini.AsyncOpenAI") as client_type:
        GeminiProvider(settings=settings)

    client_type.assert_called_once_with(
        api_key="gemini-test-key",
        base_url=("https://generativelanguage.googleapis.com/v1beta/openai/"),
        timeout=16.0,
        max_retries=3,
    )


def test_gemini_provider_rejects_missing_key() -> None:
    settings = gemini_settings(gemini_api_key=None)

    with pytest.raises(ValueError, match="provider is disabled"):
        GeminiProvider(settings=settings)
