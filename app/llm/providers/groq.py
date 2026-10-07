from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from app.core.config import Settings, get_settings
from app.llm.base import BaseLLMProvider
from app.llm.capabilities import ProviderCapabilities, STREAMING_TEXT_CAPABILITIES
from app.llm.error_mapping import map_provider_exception
from app.llm.streaming import ProviderStreamChunk, stream_openai_chat


class GroqProvider(BaseLLMProvider):
    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        settings = settings or get_settings()

        self.default_model = settings.groq_fast_model
        self.client = client or AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            timeout=settings.groq_timeout_seconds,
            max_retries=settings.groq_max_retries,
        )

    @property
    def name(self) -> str:
        return "groq"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return STREAMING_TEXT_CAPABILITIES

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> dict[str, str]:
        selected_model = model or self.default_model
        try:
            response = await self.client.chat.completions.create(
                model=selected_model,
                messages=messages,
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
    ) -> AsyncIterator[ProviderStreamChunk]:
        async for chunk in stream_openai_chat(
            client=self.client,
            provider_name=self.name,
            model=model or self.default_model,
            messages=messages,
        ):
            yield chunk

    async def close(self) -> None:
        await self.client.close()
