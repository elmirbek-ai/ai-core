from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import Any, NoReturn, cast

from openai import AsyncOpenAI
from openai.types.responses import ResponseInputParam

from app.core.config import Settings, get_settings
from app.llm.base import BaseLLMProvider
from app.llm.capabilities import ProviderCapabilities, messages_contain_images
from app.llm.error_mapping import map_provider_exception
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.streaming import ProviderStreamChunk
from app.schemas.chat import ChatRequest

OFFICIAL_BASE_URL = "https://api.openai.com/v1"


def _safe_error(error: Exception) -> LLMProviderError:
    # Preserve domain classification while sanitizing even injected/callback
    # domain errors; SDK failures use the existing shared mapping.
    for category in (
        LLMAuthenticationError,
        LLMRateLimitError,
        LLMTimeoutError,
        LLMUpstreamError,
        LLMProviderError,
    ):
        if isinstance(error, category):
            return category("OpenAI provider request failed")
    return map_provider_exception(error, "openai")


def as_responses_input(
    messages: list[dict[str, Any]], *, images: bool
) -> ResponseInputParam:
    """Reuse API validation; preserve message/part order without fetching images."""
    validated = ChatRequest.model_validate({"messages": messages})
    payload = [message.model_dump(mode="json") for message in validated.messages]
    if messages_contain_images(payload) and not images:
        raise LLMProviderError("OpenAI candidate image support is not enabled")
    converted = []
    for message in payload:
        content = message["content"]
        if isinstance(content, list):
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append({"type": "input_text", "text": part["text"]})
                else:
                    parts.append(
                        {
                            "type": "input_image",
                            "image_url": part["image_url"]["url"],
                            "detail": "auto",
                        }
                    )
            content = parts
        converted.append({"role": message["role"], "content": content})
    return cast(ResponseInputParam, converted)


def _raise_response_error(code: object) -> NoReturn:
    # Only controlled categories, never SDK messages/body/URL or repr(code).
    if code in ("invalid_api_key", "authentication_error"):
        raise LLMAuthenticationError("OpenAI authentication failed") from None
    if code == "rate_limit_exceeded":
        raise LLMRateLimitError("OpenAI rate limit reached") from None
    if code in ("timeout", "request_timeout", "vector_store_timeout"):
        raise LLMTimeoutError("OpenAI request timed out") from None
    if code == "server_error":
        raise LLMUpstreamError("OpenAI upstream request failed") from None
    raise LLMProviderError("OpenAI provider request failed") from None


def _require_completed(response: Any) -> None:
    error = getattr(response, "error", None)
    status = getattr(response, "status", None)
    if error is not None or status == "failed":
        _raise_response_error(getattr(error, "code", None))
    if status != "completed":
        raise LLMProviderError("OpenAI response did not complete") from None


class OpenAIProvider(BaseLLMProvider):
    """Responses API candidate; registry registration is not routing admission."""

    def __init__(
        self, settings: Settings | None = None, client: AsyncOpenAI | None = None
    ) -> None:
        settings = settings or get_settings()
        if settings.openai_api_key is None:
            raise ValueError("OpenAI candidate is disabled: API key is missing")
        self.model = settings.openai_model
        self._supports_images = settings.openai_supports_images
        if client is not None:
            self.client = client
        else:
            try:
                self.client = AsyncOpenAI(
                    api_key=settings.openai_api_key.get_secret_value(),
                    # Override the SDK's environment base-URL fallback: direct
                    # means the official endpoint, not an implicit proxy.
                    base_url=OFFICIAL_BASE_URL,
                    timeout=settings.openai_timeout_seconds,
                    max_retries=settings.openai_max_retries,
                )
            except Exception:
                raise LLMProviderError("OpenAI client initialization failed") from None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(images=self._supports_images, streaming=True)

    def _selected_model(self, model: str | None) -> str:
        selected = model or self.model
        if (
            not selected
            or "://" in selected
            or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,199}", selected) is None
        ):
            raise LLMProviderError(
                "OpenAI evaluation model is missing or invalid"
            ) from None
        return selected

    async def chat(
        self, messages: list[dict[str, Any]], model: str | None = None
    ) -> dict[str, str]:
        try:
            selected = self._selected_model(model)
            response = await self.client.responses.create(
                model=selected,
                input=as_responses_input(messages, images=self._supports_images),
                store=False,
            )
            _require_completed(response)
            content = response.output_text
            if not isinstance(content, str):
                raise TypeError("Invalid OpenAI text output")
            # Do not copy untrusted upstream model strings into provenance/logs.
            return {"provider": self.name, "model": selected, "content": content}
        except Exception as error:
            raise _safe_error(error) from None

    async def stream_chat(
        self, messages: list[dict[str, Any]], model: str | None = None
    ) -> AsyncGenerator[ProviderStreamChunk]:
        stream = None
        try:
            selected = self._selected_model(model)
            stream = await self.client.responses.create(
                model=selected,
                input=as_responses_input(messages, images=self._supports_images),
                stream=True,
                store=False,
            )
            async for event in stream:
                kind = getattr(event, "type", None)
                if not isinstance(kind, str) or not kind:
                    raise TypeError("Invalid OpenAI stream event")
                if kind == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if not isinstance(delta, str):
                        raise TypeError("Invalid OpenAI text delta")
                    if delta:
                        yield ProviderStreamChunk(self.name, selected, delta)
                elif kind == "response.completed":
                    _require_completed(getattr(event, "response", None))
                    return
                elif kind == "response.failed":
                    _raise_response_error(
                        getattr(
                            getattr(getattr(event, "response", None), "error", None),
                            "code",
                            None,
                        )
                    )
                elif kind == "response.incomplete":
                    raise LLMProviderError("OpenAI response did not complete") from None
                elif kind == "error":
                    _raise_response_error(getattr(event, "code", None))
                # Lifecycle/refusal/reasoning/tool events never become text.
            raise LLMUpstreamError("OpenAI stream ended before completion") from None
        except Exception as error:
            raise _safe_error(error) from None
        finally:
            if stream is not None:
                # Preserve cancellation/original outcome; registry-owned
                # client shutdown is the final cleanup boundary.
                with suppress(Exception):
                    await stream.close()

    async def close(self) -> None:
        try:
            await self.client.close()
        except Exception as error:
            raise _safe_error(error) from None
