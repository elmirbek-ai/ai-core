from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from app.core.config import Settings, get_settings
from app.llm.base import BaseLLMProvider
from app.llm.capabilities import STREAMING_IMAGE_CAPABILITIES, ProviderCapabilities
from app.llm.error_mapping import map_provider_exception
from app.llm.streaming import (
    ProviderStreamChunk,
    as_openai_messages,
    stream_openai_chat,
)


class GeminiProvider(BaseLLMProvider):
    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        settings = settings or get_settings()
        if not settings.gemini_api_key:
            raise ValueError("Gemini provider is disabled: API key is missing")

        self.model = settings.gemini_model
        self.client = client or AsyncOpenAI(
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
            timeout=settings.gemini_timeout_seconds,
            max_retries=settings.gemini_max_retries,
        )

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return STREAMING_IMAGE_CAPABILITIES

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> dict[str, str]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=as_openai_messages(messages),
            )
        except Exception as exc:
            raise map_provider_exception(exc, self.name) from None

        return {
            "provider": self.name,
            "model": response.model,
            "content": response.choices[0].message.content or "",
        }

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> AsyncGenerator[ProviderStreamChunk]:
        async for chunk in stream_openai_chat(
            client=self.client,
            provider_name=self.name,
            model=self.model,
            messages=messages,
        ):
            yield chunk

    async def close(self) -> None:
        await self.client.close()
