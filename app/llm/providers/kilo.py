import asyncio
from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.llm.base import BaseLLMProvider
from app.llm.error_mapping import map_httpx_provider_exception
from app.llm.exceptions import (
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)


RETRYABLE_ERRORS = (
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
RETRY_BASE_DELAY_SECONDS = 0.1


class KiloProvider(BaseLLMProvider):
    """Opt-in client for Kilo's anonymous, shared-capacity gateway."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = settings or get_settings()
        if not settings.kilo_enabled:
            raise ValueError("Kilo provider is disabled")

        self.model = settings.kilo_general_model
        self.max_retries = settings.kilo_max_retries
        self.endpoint = (
            f"{settings.kilo_base_url.rstrip('/')}/chat/completions"
        )
        self.headers = {"Content-Type": "application/json"}
        self.client = client or httpx.AsyncClient(
            timeout=settings.kilo_timeout_seconds,
        )

    @property
    def name(self) -> str:
        return "kilo"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> dict[str, str]:
        selected_model = model or self.model
        response_data = await self._request_with_retries(
            messages=messages,
            model=selected_model,
        )
        try:
            actual_model = response_data["model"]
            content = response_data["choices"][0]["message"]["content"] or ""
            if not isinstance(actual_model, str) or not isinstance(content, str):
                raise TypeError("Invalid Kilo response fields")
        except Exception as exc:
            raise map_httpx_provider_exception(exc, self.name) from None

        return {
            "provider": self.name,
            "model": actual_model,
            "content": content,
        }

    async def _request_with_retries(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.post(
                    self.endpoint,
                    headers=self.headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise TypeError("Invalid Kilo response")
                return data
            except Exception as exc:
                mapped_error = map_httpx_provider_exception(exc, self.name)
                can_retry = isinstance(mapped_error, RETRYABLE_ERRORS)
                if not can_retry or attempt == self.max_retries:
                    raise mapped_error from None
                await asyncio.sleep(RETRY_BASE_DELAY_SECONDS * (2**attempt))

        raise LLMProviderError("Kilo provider request failed")

    async def close(self) -> None:
        await self.client.aclose()
