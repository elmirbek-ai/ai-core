import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.core.config import Settings
from app.llm.base import BaseLLMProvider
from app.llm.providers.groq import GroqProvider


def test_groq_provider_contract() -> None:
    create = AsyncMock(
        return_value=SimpleNamespace(
            model="openai/gpt-oss-20b",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="test response"),
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
    provider = GroqProvider(client=client)

    async def exercise_provider() -> dict[str, str]:
        result = await provider.chat(
            [{"role": "user", "content": "Hello"}],
        )
        await provider.close()
        return result

    result = asyncio.run(exercise_provider())

    assert isinstance(provider, BaseLLMProvider)
    assert result == {
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "content": "test response",
    }
    create.assert_awaited_once()
    close.assert_awaited_once()


def test_groq_client_uses_timeout_and_retries() -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        groq_timeout_seconds=12.5,
        groq_max_retries=4,
    )

    with patch("app.llm.providers.groq.AsyncOpenAI") as client_type:
        GroqProvider(settings=settings)

    client_type.assert_called_once_with(
        api_key="test-key",
        base_url="https://api.groq.com/openai/v1",
        timeout=12.5,
        max_retries=4,
    )


def test_groq_config_safe_defaults() -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
    )

    assert settings.groq_timeout_seconds == 30.0
    assert settings.groq_max_retries == 2
    assert settings.groq_fast_model == "openai/gpt-oss-20b"
    assert settings.groq_reasoning_model == "openai/gpt-oss-120b"


def test_groq_provider_uses_selected_model() -> None:
    create = AsyncMock(
        return_value=SimpleNamespace(
            model="openai/gpt-oss-120b",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="reasoned response"),
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
    provider = GroqProvider(client=client)

    result = asyncio.run(
        provider.chat(
            [{"role": "user", "content": "Analyze"}],
            model="openai/gpt-oss-120b",
        )
    )

    create.assert_awaited_once_with(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "Analyze"}],
    )
    assert result["model"] == "openai/gpt-oss-120b"


def test_legacy_groq_provider_is_removed() -> None:
    legacy_provider = Path(__file__).parents[1] / "app" / "llm" / "groq.py"

    assert not legacy_provider.exists()
