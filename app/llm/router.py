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
    ) -> None:
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.gemini_provider = gemini_provider
        self.cloudflare_provider = cloudflare_provider

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
                (self.cloudflare_provider, None),
                (self.primary_provider, None),
            ]
        if task == TaskType.MULTIMODAL:
            return [
                (self.gemini_provider, None),
                (self.fallback_provider, None),
                (self.primary_provider, None),
            ]
        if task in REASONING_TASKS:
            return [
                (self.primary_provider, model),
                (self.cloudflare_provider, None),
                (self.fallback_provider, None),
            ]
        if task in FAST_TASKS:
            return [
                (self.primary_provider, model),
                (self.fallback_provider, None),
                (self.cloudflare_provider, None),
            ]
        raise ValueError(f"Unsupported task type: {task}")

    async def _run_chain(
        self,
        messages: list[dict[str, Any]],
        chain: list[tuple[BaseLLMProvider | None, str | None]],
    ) -> dict[str, str]:
        provider_calls = [
            (provider, provider_model)
            for provider, provider_model in chain
            if provider is not None
        ]
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
