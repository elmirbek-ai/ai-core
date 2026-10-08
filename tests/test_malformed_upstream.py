import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import Settings
from app.llm.base import BaseLLMProvider
from app.llm.capabilities import ProviderCapabilities
from app.llm.exceptions import LLMProviderError
from app.llm.providers.cloudflare import CloudflareProvider
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.groq import GroqProvider
from app.llm.providers.llm7 import LLM7Provider
from app.llm.providers.ollama import OllamaProvider
from app.llm.providers.openrouter import OpenRouterProvider
from app.llm.router import LLMRouter
from app.llm.streaming import ProviderStreamChunk, stream_openai_chat

MESSAGES = [{"role": "user", "content": "private prompt"}]


def _openai_client(response: object) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        ),
        close=AsyncMock(),
    )


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
@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(model="private-model", choices=[]),
        SimpleNamespace(
            model="private-model",
            choices=[SimpleNamespace(message=SimpleNamespace(content={"raw": "body"}))],
        ),
    ],
)
def test_openai_compatible_malformed_response_is_sanitized(
    provider_type,
    response: object,
) -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openrouter_api_key="test-key",
        gemini_api_key="test-key",
        cloudflare_api_token="test-token",
        cloudflare_account_id="test-account",
        llm7_enabled=True,
        llm7_api_key="test-key",
    )
    provider = provider_type(settings=settings, client=_openai_client(response))

    with pytest.raises(LLMProviderError) as captured:
        asyncio.run(provider.chat(MESSAGES))

    message = str(captured.value)
    assert message == f"{provider.name} provider request failed"
    assert "private-model" not in message
    assert "body" not in message
    assert "private prompt" not in message


def test_ollama_malformed_response_is_sanitized() -> None:
    request = httpx.Request("POST", "https://ollama.invalid/api/chat")
    response = httpx.Response(
        200,
        json={"unexpected": "private raw body"},
        request=request,
    )
    client = SimpleNamespace(post=AsyncMock(return_value=response), aclose=AsyncMock())
    provider = OllamaProvider(
        settings=Settings(
            _env_file=None,
            groq_api_key="test-key",
            ollama_api_key="test-key",
            ollama_base_url="https://ollama.invalid",
            ollama_max_retries=0,
        ),
        client=client,
    )

    with pytest.raises(LLMProviderError) as captured:
        asyncio.run(provider.chat(MESSAGES))

    assert str(captured.value) == "ollama provider request failed"
    assert "private raw body" not in str(captured.value)


class FakeStream:
    def __init__(self, chunks: list[object]) -> None:
        self._chunks = chunks
        self.aclose = AsyncMock()

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


def test_stream_skips_empty_chunks_and_closes_upstream() -> None:
    stream = FakeStream(
        [
            SimpleNamespace(model="model", choices=[]),
            SimpleNamespace(
                model="model",
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None))],
            ),
            SimpleNamespace(
                model="actual-model",
                choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))],
            ),
        ]
    )
    client = _openai_client(stream)

    async def exercise() -> list[ProviderStreamChunk]:
        return [
            chunk
            async for chunk in stream_openai_chat(
                client=client,
                provider_name="groq",
                model="configured-model",
                messages=MESSAGES,
            )
        ]

    chunks = asyncio.run(exercise())

    assert chunks == [ProviderStreamChunk("groq", "actual-model", "ok")]
    stream.aclose.assert_awaited_once()


def test_malformed_stream_chunk_is_sanitized_and_closed() -> None:
    stream = FakeStream([SimpleNamespace(raw="private upstream body")])
    client = _openai_client(stream)

    async def exercise() -> None:
        with pytest.raises(LLMProviderError) as captured:
            async for _ in stream_openai_chat(
                client=client,
                provider_name="groq",
                model="model",
                messages=MESSAGES,
            ):
                pass
        assert str(captured.value) == "groq provider request failed"
        assert "private upstream body" not in str(captured.value)

    asyncio.run(exercise())
    stream.aclose.assert_awaited_once()


class EmptyStreamingProvider(BaseLLMProvider):
    def __init__(self, name: str, content: str | None = None) -> None:
        self._name = name
        self.content = content
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True)

    async def chat(self, messages, model=None):
        del messages, model
        raise AssertionError("non-stream path must not be used")

    async def stream_chat(self, messages, model=None):
        del messages, model
        self.calls += 1
        if self.content is not None:
            yield ProviderStreamChunk(self.name, "model", self.content)

    async def close(self) -> None:
        return None


def test_empty_stream_falls_back_without_leaking_failed_provider_meta() -> None:
    primary = EmptyStreamingProvider("groq")
    fallback = EmptyStreamingProvider("openrouter", "ok")
    router = LLMRouter(primary, fallback)

    async def exercise() -> list:
        return [event async for event in router.stream_chat(MESSAGES)]

    events = asyncio.run(exercise())

    assert [event.event for event in events] == ["meta", "delta", "done"]
    assert events[0].data["provider"] == "openrouter"
    assert primary.calls == 1
    assert fallback.calls == 1
