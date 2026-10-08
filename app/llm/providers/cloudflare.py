from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from app.core.config import Settings, get_settings
from app.llm.base import BaseLLMProvider
from app.llm.capabilities import STREAMING_TEXT_CAPABILITIES, ProviderCapabilities
from app.llm.error_mapping import map_provider_exception
from app.llm.exceptions import LLMProviderError
from app.llm.streaming import (
    ProviderStreamChunk,
    as_openai_messages,
    stream_openai_chat,
)


class CloudflareProvider(BaseLLMProvider):
    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        settings = settings or get_settings()
        if not settings.cloudflare_api_token:
            raise ValueError("Cloudflare provider is disabled: API token is missing")
        if not settings.cloudflare_account_id:
            raise ValueError("Cloudflare provider is disabled: account ID is missing")

        self.model = settings.cloudflare_model
        base_url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{settings.cloudflare_account_id}/ai/v1"
        )
        if client is not None:
            self.client = client
        else:
            try:
                self.client = AsyncOpenAI(
                    api_key=settings.cloudflare_api_token,
                    base_url=base_url,
                    timeout=settings.cloudflare_timeout_seconds,
                    max_retries=settings.cloudflare_max_retries,
                )
            except Exception:
                raise LLMProviderError(
                    "Cloudflare client initialization failed",
                ) from None

    @property
    def name(self) -> str:
        return "cloudflare"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return STREAMING_TEXT_CAPABILITIES

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

        try:
            actual_model = response.model
            content = response.choices[0].message.content or ""
            if not isinstance(actual_model, str) or not isinstance(content, str):
                raise TypeError("Invalid Cloudflare response fields")
        except Exception as exc:
            raise map_provider_exception(exc, self.name) from None

        return {
            "provider": self.name,
            "model": actual_model,
            "content": content,
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
