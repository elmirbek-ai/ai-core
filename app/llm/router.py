import asyncio
from collections.abc import Callable
import logging
import time
from typing import Any

from app.llm.base import BaseLLMProvider
from app.llm.budget import RequestBudgetPolicy
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.health import ProviderHealthManager
from app.llm.model_router import FAST_TASKS, REASONING_TASKS
from app.llm.task import TaskType


logger = logging.getLogger(__name__)

RECOVERABLE_PROVIDER_ERRORS = (
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)


class _RequestBudgetExceeded(Exception):
    pass


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
        llm7_provider: BaseLLMProvider | None = None,
        llm7_general_model: str | None = None,
        llm7_reasoning_model: str | None = None,
        llm7_code_model: str | None = None,
        health_manager: ProviderHealthManager | None = None,
        budget_policy: RequestBudgetPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
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
        self.llm7_provider = llm7_provider
        self.llm7_general_model = llm7_general_model
        self.llm7_reasoning_model = llm7_reasoning_model
        self.llm7_code_model = llm7_code_model
        self.health_manager = health_manager
        self.budget_policy = budget_policy
        self._clock = clock

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        task: TaskType = TaskType.GENERAL,
    ) -> dict[str, str]:
        chain = self._provider_chain(task=task, model=model)
        budget_seconds = (
            self.budget_policy.resolve(task)
            if self.budget_policy is not None
            else None
        )
        deadline = (
            self._clock() + budget_seconds
            if budget_seconds is not None
            else None
        )
        return await self._run_chain(
            messages=messages,
            chain=chain,
            task=task,
            deadline=deadline,
        )

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
                (self.llm7_provider, self.llm7_code_model),
            ]
        if task in REASONING_TASKS:
            return [
                (self.primary_provider, model),
                (self.cloudflare_provider, None),
                (self.ollama_provider, None),
                (self.fallback_provider, None),
                (self.llm7_provider, self.llm7_reasoning_model),
            ]
        if task in FAST_TASKS:
            return [
                (self.primary_provider, model),
                (self.fallback_provider, None),
                (self.cloudflare_provider, None),
                (self.ollama_provider, None),
                (self.kilo_provider, self.kilo_general_model),
                (self.llm7_provider, self.llm7_general_model),
            ]
        raise ValueError(f"Unsupported task type: {task}")

    async def _run_chain(
        self,
        messages: list[dict[str, Any]],
        chain: list[tuple[BaseLLMProvider | None, str | None]],
        task: TaskType,
        deadline: float | None,
    ) -> dict[str, str]:
        provider_calls: list[tuple[BaseLLMProvider, str | None]] = []
        seen_provider_ids: set[int] = set()
        for provider, provider_model in chain:
            if provider is None or id(provider) in seen_provider_ids:
                continue
            seen_provider_ids.add(id(provider))
            provider_calls.append((provider, provider_model))

        if self.health_manager is not None:
            available_calls: list[
                tuple[BaseLLMProvider, str | None]
            ] = []
            for provider, provider_model in provider_calls:
                if await self.health_manager.is_available(provider.name):
                    available_calls.append((provider, provider_model))
                else:
                    logger.warning(
                        "Provider %s skipped because circuit open",
                        provider.name,
                    )
            provider_calls = available_calls

        recoverable_failures = 0

        for index, (provider, provider_model) in enumerate(provider_calls):
            remaining = None
            if deadline is not None:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    logger.warning(
                        "Request budget exhausted for task %s",
                        task.value,
                    )
                    raise LLMTimeoutError(
                        "LLM request budget exhausted",
                    ) from None
                logger.debug(
                    "Provider %s attempt limited by remaining budget for task %s",
                    provider.name,
                    task.value,
                )
            try:
                result = await self._call_provider(
                    provider=provider,
                    provider_model=provider_model,
                    messages=messages,
                    remaining=remaining,
                )
                if self.health_manager is not None:
                    await self.health_manager.record_success(provider.name)
                return result
            except _RequestBudgetExceeded:
                timeout_error = LLMTimeoutError(
                    "LLM request budget exhausted",
                )
                if self.health_manager is not None:
                    await self.health_manager.record_failure(
                        provider.name,
                        timeout_error,
                    )
                logger.warning(
                    "Request budget exhausted for task %s while calling provider %s",
                    task.value,
                    provider.name,
                )
                raise timeout_error from None
            except LLMAuthenticationError:
                raise
            except RECOVERABLE_PROVIDER_ERRORS as error:
                if self.health_manager is not None:
                    await self.health_manager.record_failure(
                        provider.name,
                        error,
                    )
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

    async def _call_provider(
        self,
        provider: BaseLLMProvider,
        provider_model: str | None,
        messages: list[dict[str, Any]],
        remaining: float | None,
    ) -> dict[str, str]:
        async def invoke() -> dict[str, str]:
            if provider_model is None:
                return await provider.chat(messages)
            return await provider.chat(messages, model=provider_model)

        if remaining is None:
            return await invoke()
        try:
            async with asyncio.timeout(remaining):
                return await invoke()
        except TimeoutError:
            raise _RequestBudgetExceeded from None
