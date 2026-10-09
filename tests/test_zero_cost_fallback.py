import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.llm.capabilities import ProviderCapabilities
from app.llm.exceptions import LLMRateLimitError
from app.llm.registry import create_provider_registry
from app.llm.router import LLMRouter


def test_quota_failure_falls_back_only_to_registered_free_route(monkeypatch):
    groq = SimpleNamespace(
        name="groq",
        chat=AsyncMock(side_effect=LLMRateLimitError("quota exhausted")),
        close=AsyncMock(),
    )
    fallback = SimpleNamespace(
        name="openrouter",
        chat=AsyncMock(
            return_value={
                "provider": "openrouter",
                "model": "openrouter/free",
                "content": "safe",
            }
        ),
        close=AsyncMock(),
    )
    monkeypatch.setattr("app.llm.registry.GroqProvider", lambda settings: groq)
    monkeypatch.setattr(
        "app.llm.registry.OpenRouterProvider", lambda settings: fallback
    )
    settings = Settings(
        _env_file=None, groq_api_key="test-key", openrouter_api_key="test-key"
    )

    async def exercise():
        registry = await create_provider_registry(settings)
        router = LLMRouter(
            registry.require_enabled("groq"), registry.get_enabled("openrouter")
        )
        result = await router.chat([{"role": "user", "content": "hello"}])
        assert result["model"] == "openrouter/free"
        await registry.close()

    asyncio.run(exercise())
    fallback.chat.assert_awaited_once()
    groq.close.assert_awaited_once()
    fallback.close.assert_awaited_once()


def test_streaming_quota_failure_does_not_attempt_paid_fallback(monkeypatch):
    async def stream(messages, model=None):
        raise LLMRateLimitError("quota exhausted")
        yield

    groq = SimpleNamespace(
        name="groq",
        stream_chat=stream,
        capabilities=ProviderCapabilities(streaming=True),
        close=AsyncMock(),
    )
    paid_factory = AsyncMock()
    monkeypatch.setattr("app.llm.registry.GroqProvider", lambda settings: groq)
    monkeypatch.setattr("app.llm.registry.OpenRouterProvider", paid_factory)
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openrouter_api_key="test-key",
        openrouter_model="vendor/paid",
    )

    async def exercise():
        registry = await create_provider_registry(settings)
        router = LLMRouter(
            registry.require_enabled("groq"), registry.get_enabled("openrouter")
        )
        with pytest.raises(LLMRateLimitError):
            _ = [
                event
                async for event in router.stream_chat(
                    [{"role": "user", "content": "hello"}]
                )
            ]
        await registry.close()

    asyncio.run(exercise())
    paid_factory.assert_not_called()


def test_mixed_kilo_task_models_do_not_activate_partial_free_profile(monkeypatch):
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    factory = AsyncMock()
    monkeypatch.setattr("app.llm.registry.GroqProvider", lambda settings: groq)
    monkeypatch.setattr("app.llm.registry.KiloProvider", factory)
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        kilo_enabled=True,
        kilo_general_model="kilo-auto/free",
        kilo_code_model="kilo-auto/free",
        kilo_long_context_model="dots-studio/dots-3-note-preview:free",
    )

    async def exercise():
        registry = await create_provider_registry(settings)
        assert not registry.is_enabled("kilo")
        await registry.close()

    asyncio.run(exercise())
    factory.assert_not_called()
