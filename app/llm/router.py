import logging
from typing import Any

from app.llm.base import BaseLLMProvider
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.model_router import FAST_TASKS, REASONING_TASKS
from app.llm.task import TaskType


logger = logging.getLogger(__name__)

RECOVERABLE_PROVIDER_ERRORS = (
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)


class LLMRouter:
    def __init__(
        self,
        primary_provider: BaseLLMProvider,
        fallback_provider: BaseLLMProvider | None = None,
        gemini_provider: BaseLLMProvider | None = None,
        cloudflare_provider: BaseLLMProvider | None = None,
        ollama_provider: BaseLLMProvider | None = None,
        kilo_provider: BaseLLMProvider | None = None,
        kilo_general_model: str | None = None,
        kilo_code_model: str | None = None,
        kilo_long_context_model: str | None = None,
    ) -> None:
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.gemini_provider = gemini_provider
        self.cloudflare_provider = cloudflare_provider
        self.ollama_provider = ollama_provider
        self.kilo_provider = kilo_provider
        self.kilo_general_model = kilo_general_model
        self.kilo_code_model = kilo_code_model
        self.kilo_long_context_model = kilo_long_context_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        task: TaskType = TaskType.GENERAL,
    ) -> dict[str, str]:
        chain = self._provider_chain(task=task, model=model)
        return await self._run_chain(messages=messages, chain=chain)

    def _provider_chain(
        self,
        task: TaskType,
        model: str | None,
    ) -> list[tuple[BaseLLMProvider | None, str | None]]:
        if task == TaskType.LONG_CONTEXT:
            return [
                (self.gemini_provider, None),
                (self.fallback_provider, None),
                (self.kilo_provider, self.kilo_long_context_model),
                (self.cloudflare_provider, None),
                (self.ollama_provider, None),
                (self.primary_provider, None),
            ]
        if task == TaskType.MULTIMODAL:
            return [
                (self.gemini_provider, None),
                (self.fallback_provider, None),
                (self.primary_provider, None),
            ]
        if task == TaskType.CODE:
            return [
                (self.primary_provider, model),
                (self.cloudflare_provider, None),
                (self.ollama_provider, None),
                (self.fallback_provider, None),
                (self.kilo_provider, self.kilo_code_model),
            ]
        if task in REASONING_TASKS:
            return [
                (self.primary_provider, model),
                (self.cloudflare_provider, None),
                (self.ollama_provider, None),
                (self.fallback_provider, None),
            ]
        if task in FAST_TASKS:
            return [
                (self.primary_provider, model),
                (self.fallback_provider, None),
                (self.cloudflare_provider, None),
                (self.ollama_provider, None),
                (self.kilo_provider, self.kilo_general_model),
            ]
        raise ValueError(f"Unsupported task type: {task}")

    async def _run_chain(
        self,
        messages: list[dict[str, Any]],
        chain: list[tuple[BaseLLMProvider | None, str | None]],
    ) -> dict[str, str]:
        provider_calls: list[tuple[BaseLLMProvider, str | None]] = []
        seen_provider_ids: set[int] = set()
        for provider, provider_model in chain:
            if provider is None or id(provider) in seen_provider_ids:
                continue
            seen_provider_ids.add(id(provider))
            provider_calls.append((provider, provider_model))

        recoverable_failures = 0

        for index, (provider, provider_model) in enumerate(provider_calls):
            try:
                if provider_model is None:
                    return await provider.chat(messages)
                return await provider.chat(messages, model=provider_model)
            except LLMAuthenticationError:
                raise
            except RECOVERABLE_PROVIDER_ERRORS:
                recoverable_failures += 1
                if index == len(provider_calls) - 1:
                    if recoverable_failures == 1:
                        raise
                    raise LLMProviderError(
                        "All task providers failed",
                    ) from None

                next_provider = provider_calls[index + 1][0]
                logger.warning(
                    "Provider %s failed, falling back to provider %s",
                    provider.name,
                    next_provider.name,
                )

        raise LLMProviderError("No provider is available for task")
