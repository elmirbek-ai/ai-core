import asyncio
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from app.core.observability import error_category, log_event
from app.llm.base import BaseLLMProvider
from app.llm.budget import RequestBudgetPolicy
from app.llm.capabilities import ProviderCapabilities, messages_contain_images
from app.llm.concurrency import (
    ProviderConcurrencyManager,
    ProviderConcurrencyTimeout,
)
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.health import ProviderHealthManager
from app.llm.model_router import FAST_TASKS, REASONING_TASKS
from app.llm.streaming import ProviderStreamChunk, StreamEvent
from app.llm.task import TaskType
from app.llm.telemetry import LLMTelemetry

logger = logging.getLogger(__name__)

RECOVERABLE_PROVIDER_ERRORS = (
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)


class _RequestBudgetExceeded(Exception):
    pass


@dataclass(slots=True)
class _RequestExecutionState:
    failed_attempts: int = 0
    selected_provider: str | None = None


@dataclass(slots=True)
class _StreamExecutionState:
    selected_provider: str | None = None
    first_token_time: float | None = None
    selected_attempt_started: float | None = None


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
        telemetry: LLMTelemetry | None = None,
        concurrency_manager: ProviderConcurrencyManager | None = None,
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
        self.telemetry = telemetry
        self.concurrency_manager = concurrency_manager
        self._clock = clock

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        task: TaskType = TaskType.GENERAL,
    ) -> dict[str, str]:
        request_started = self._clock()
        execution = _RequestExecutionState()
        try:
            requires_images = messages_contain_images(messages)
            if requires_images and task != TaskType.MULTIMODAL:
                raise LLMProviderError(
                    "Image content requires a multimodal task",
                ) from None
            chain = self._provider_chain(
                task=task,
                model=model,
                requires_images=requires_images,
            )
            budget_seconds = (
                self.budget_policy.resolve(task)
                if self.budget_policy is not None
                else None
            )
            deadline = (
                request_started + budget_seconds if budget_seconds is not None else None
            )
            result = await self._run_chain(
                messages=messages,
                chain=chain,
                task=task,
                deadline=deadline,
                execution=execution,
            )
        except Exception as error:
            if self.telemetry is not None:
                await self.telemetry.record_request_failure(
                    task=task,
                    error=error,
                    latency_seconds=self._clock() - request_started,
                    fallback_depth=execution.failed_attempts,
                )
            raise

        if self.telemetry is not None:
            await self.telemetry.record_request_success(
                task=task,
                selected_provider=(execution.selected_provider or result["provider"]),
                latency_seconds=self._clock() - request_started,
                fallback_depth=execution.failed_attempts,
            )
        return result

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        task: TaskType = TaskType.GENERAL,
    ) -> AsyncGenerator[StreamEvent]:
        request_started = self._clock()
        execution = _StreamExecutionState()
        success = False
        cancelled = False
        stream_error: BaseException | None = None
        log_event(logger, "stream_started", task=task.value, streaming=True)
        try:
            requires_images = messages_contain_images(messages)
            if requires_images and task != TaskType.MULTIMODAL:
                raise LLMProviderError(
                    "Image content requires a multimodal task",
                ) from None
            chain = self._provider_chain(
                task=task,
                model=model,
                requires_images=requires_images,
            )
            budget_seconds = (
                self.budget_policy.resolve(task)
                if self.budget_policy is not None
                else None
            )
            deadline = (
                request_started + budget_seconds if budget_seconds is not None else None
            )
            stream = self._run_stream_chain(
                messages=messages,
                chain=chain,
                task=task,
                deadline=deadline,
                execution=execution,
            )
            async with aclosing(stream):
                async for event in stream:
                    yield event
            success = True
        except (asyncio.CancelledError, GeneratorExit):
            cancelled = True
            raise
        except Exception as error:
            stream_error = error
            raise
        finally:
            duration = self._clock() - request_started
            log_event(
                logger,
                "stream_cancelled"
                if cancelled
                else ("stream_completed" if success else "stream_failed"),
                task=task.value,
                provider=execution.selected_provider,
                streaming=True,
                duration_ms=max(0.0, duration) * 1000,
                error_category="cancelled"
                if cancelled
                else (error_category(stream_error) if stream_error else None),
            )
            if self.telemetry is not None:
                first_token_latency = (
                    execution.first_token_time - execution.selected_attempt_started
                    if execution.first_token_time is not None
                    and execution.selected_attempt_started is not None
                    else None
                )
                await self.telemetry.record_stream_request(
                    task=task,
                    selected_provider=execution.selected_provider,
                    success=success,
                    duration_seconds=duration,
                    time_to_first_token_seconds=first_token_latency,
                    cancelled=cancelled,
                )

    def _provider_chain(
        self,
        task: TaskType,
        model: str | None,
        requires_images: bool = False,
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
            if requires_images:
                return [
                    (provider, None)
                    for provider in (
                        self.gemini_provider,
                        self.fallback_provider,
                    )
                    if self._supports_images(provider)
                ]
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

    @staticmethod
    def _supports_images(provider: BaseLLMProvider | None) -> bool:
        if provider is None:
            return False
        capabilities = getattr(provider, "capabilities", None)
        return isinstance(capabilities, ProviderCapabilities) and capabilities.images

    @staticmethod
    def _supports_streaming(provider: BaseLLMProvider | None) -> bool:
        if provider is None:
            return False
        capabilities = getattr(provider, "capabilities", None)
        return isinstance(capabilities, ProviderCapabilities) and capabilities.streaming

    async def _run_stream_chain(
        self,
        messages: list[dict[str, Any]],
        chain: list[tuple[BaseLLMProvider | None, str | None]],
        task: TaskType,
        deadline: float | None,
        execution: _StreamExecutionState,
    ) -> AsyncGenerator[StreamEvent]:
        provider_calls: list[tuple[BaseLLMProvider, str | None]] = []
        seen_provider_ids: set[int] = set()
        for provider, provider_model in chain:
            if (
                provider is None
                or id(provider) in seen_provider_ids
                or not self._supports_streaming(provider)
            ):
                continue
            seen_provider_ids.add(id(provider))
            provider_calls.append((provider, provider_model))

        last_recoverable_error: Exception | None = None
        recoverable_failures = 0
        for index, (provider, provider_model) in enumerate(provider_calls):
            if (
                self.health_manager is not None
                and not await self.health_manager.is_available(provider.name)
            ):
                if self.telemetry is not None:
                    await self.telemetry.record_circuit_skip(provider.name)
                log_event(
                    logger,
                    "provider_circuit_skipped",
                    provider=provider.name,
                    task=task.value,
                    error_category="circuit_skip",
                )
                continue

            remaining = self._remaining_budget(deadline, task)
            emitted_delta = False
            attempt_started: float | None = None
            try:
                async with self._provider_slot(provider.name, remaining):
                    provider_remaining = self._remaining_budget(deadline, task)
                    attempt_started = self._clock()
                    async with self._observe_attempt(
                        provider.name,
                        provider_model,
                        task,
                        streaming=True,
                        fallback_depth=index,
                    ):
                        stream = self._stream_provider(
                            provider=provider,
                            provider_model=provider_model,
                            messages=messages,
                            remaining=provider_remaining,
                        )
                        async with aclosing(stream):
                            async for chunk in stream:
                                if not chunk.content:
                                    continue
                                if not emitted_delta:
                                    emitted_delta = True
                                    execution.selected_provider = provider.name
                                    execution.first_token_time = self._clock()
                                    execution.selected_attempt_started = attempt_started
                                    log_event(
                                        logger,
                                        "stream_first_token",
                                        provider=provider.name,
                                        task=task.value,
                                        streaming=True,
                                        fallback_depth=index,
                                        ttft_ms=max(
                                            0.0,
                                            execution.first_token_time
                                            - attempt_started,
                                        )
                                        * 1000,
                                    )
                                    yield StreamEvent(
                                        event="meta",
                                        data={
                                            "provider": chunk.provider,
                                            "model": chunk.model,
                                        },
                                    )
                                yield StreamEvent(
                                    event="delta",
                                    data={"content": chunk.content},
                                )
                        if not emitted_delta:
                            raise LLMUpstreamError(
                                "Provider stream ended without content",
                            ) from None
                if self.health_manager is not None:
                    await self.health_manager.record_success(provider.name)
                yield StreamEvent(event="done", data={})
                return
            except ProviderConcurrencyTimeout:
                raise LLMTimeoutError(
                    "LLM request budget exhausted",
                ) from None
            except _RequestBudgetExceeded:
                timeout_error = LLMTimeoutError(
                    "LLM request budget exhausted",
                )
                if self.health_manager is not None:
                    await self.health_manager.record_failure(
                        provider.name,
                        timeout_error,
                    )
                raise timeout_error from None
            except RECOVERABLE_PROVIDER_ERRORS as error:
                if self.health_manager is not None:
                    await self.health_manager.record_failure(
                        provider.name,
                        error,
                    )
                if emitted_delta:
                    raise
                recoverable_failures += 1
                last_recoverable_error = error
                if index < len(provider_calls) - 1:
                    logger.warning(
                        "Provider %s stream failed before first token; falling back",
                        provider.name,
                    )
        if last_recoverable_error is not None:
            if recoverable_failures == 1:
                raise last_recoverable_error
            raise LLMProviderError("All streaming providers failed") from None
        raise LLMProviderError("No streaming provider is available") from None

    def _remaining_budget(
        self,
        deadline: float | None,
        task: TaskType,
    ) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - self._clock()
        if remaining <= 0:
            logger.warning("Request budget exhausted for task %s", task.value)
            raise LLMTimeoutError("LLM request budget exhausted") from None
        return remaining

    async def _stream_provider(
        self,
        *,
        provider: BaseLLMProvider,
        provider_model: str | None,
        messages: list[dict[str, Any]],
        remaining: float | None,
    ) -> AsyncGenerator[ProviderStreamChunk]:
        async def chunks() -> AsyncGenerator[ProviderStreamChunk]:
            stream = provider.stream_chat(
                messages,
                model=provider_model,
            )
            async with aclosing(stream):
                async for chunk in stream:
                    yield chunk

        if remaining is None:
            async for chunk in chunks():
                yield chunk
            return
        try:
            async with asyncio.timeout(remaining):
                async for chunk in chunks():
                    yield chunk
        except TimeoutError:
            raise _RequestBudgetExceeded from None

    async def _run_chain(
        self,
        messages: list[dict[str, Any]],
        chain: list[tuple[BaseLLMProvider | None, str | None]],
        task: TaskType,
        deadline: float | None,
        execution: _RequestExecutionState,
    ) -> dict[str, str]:
        provider_calls: list[tuple[BaseLLMProvider, str | None]] = []
        seen_provider_ids: set[int] = set()
        for provider, provider_model in chain:
            if provider is None or id(provider) in seen_provider_ids:
                continue
            seen_provider_ids.add(id(provider))
            provider_calls.append((provider, provider_model))

        recoverable_failures = 0
        last_recoverable_error: Exception | None = None

        for index, (provider, provider_model) in enumerate(provider_calls):
            if (
                self.health_manager is not None
                and not await self.health_manager.is_available(provider.name)
            ):
                if self.telemetry is not None:
                    await self.telemetry.record_circuit_skip(provider.name)
                log_event(
                    logger,
                    "provider_circuit_skipped",
                    provider=provider.name,
                    task=task.value,
                    error_category="circuit_skip",
                )
                continue

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
            attempt_started: float | None = None
            try:
                async with self._provider_slot(provider.name, remaining):
                    provider_remaining = remaining
                    if deadline is not None:
                        provider_remaining = deadline - self._clock()
                        if provider_remaining <= 0:
                            raise ProviderConcurrencyTimeout from None
                    attempt_started = self._clock()
                    async with self._observe_attempt(
                        provider.name,
                        provider_model,
                        task,
                        streaming=False,
                        fallback_depth=index,
                    ):
                        result = await self._call_provider(
                            provider=provider,
                            provider_model=provider_model,
                            messages=messages,
                            remaining=provider_remaining,
                        )
                attempt_latency = self._clock() - attempt_started
                if self.telemetry is not None:
                    await self.telemetry.record_provider_success(
                        provider_name=provider.name,
                        latency_seconds=attempt_latency,
                    )
                if self.health_manager is not None:
                    await self.health_manager.record_success(provider.name)
                execution.selected_provider = provider.name
                return result
            except ProviderConcurrencyTimeout:
                logger.warning(
                    "Request budget exhausted waiting for provider %s concurrency slot",
                    provider.name,
                )
                raise LLMTimeoutError(
                    "LLM request budget exhausted",
                ) from None
            except _RequestBudgetExceeded:
                assert attempt_started is not None
                attempt_latency = self._clock() - attempt_started
                timeout_error = LLMTimeoutError(
                    "LLM request budget exhausted",
                )
                if self.health_manager is not None:
                    await self.health_manager.record_failure(
                        provider.name,
                        timeout_error,
                    )
                if self.telemetry is not None:
                    await self.telemetry.record_provider_failure(
                        provider_name=provider.name,
                        error=timeout_error,
                        latency_seconds=attempt_latency,
                    )
                execution.failed_attempts += 1
                logger.warning(
                    "Request budget exhausted for task %s while calling provider %s",
                    task.value,
                    provider.name,
                )
                raise timeout_error from None
            except LLMAuthenticationError as error:
                assert attempt_started is not None
                attempt_latency = self._clock() - attempt_started
                if self.telemetry is not None:
                    await self.telemetry.record_provider_failure(
                        provider_name=provider.name,
                        error=error,
                        latency_seconds=attempt_latency,
                    )
                execution.failed_attempts += 1
                raise
            except RECOVERABLE_PROVIDER_ERRORS as error:
                assert attempt_started is not None
                attempt_latency = self._clock() - attempt_started
                if self.health_manager is not None:
                    await self.health_manager.record_failure(
                        provider.name,
                        error,
                    )
                if self.telemetry is not None:
                    await self.telemetry.record_provider_failure(
                        provider_name=provider.name,
                        error=error,
                        latency_seconds=attempt_latency,
                    )
                execution.failed_attempts += 1
                recoverable_failures += 1
                last_recoverable_error = error
                if index < len(provider_calls) - 1:
                    next_provider = provider_calls[index + 1][0]
                    logger.warning(
                        "Provider %s failed, falling back to provider %s",
                        provider.name,
                        next_provider.name,
                    )
            except LLMProviderError as error:
                assert attempt_started is not None
                attempt_latency = self._clock() - attempt_started
                if self.telemetry is not None:
                    await self.telemetry.record_provider_failure(
                        provider_name=provider.name,
                        error=error,
                        latency_seconds=attempt_latency,
                    )
                execution.failed_attempts += 1
                raise
            except Exception as error:
                assert attempt_started is not None
                attempt_latency = self._clock() - attempt_started
                if self.telemetry is not None:
                    await self.telemetry.record_provider_failure(
                        provider_name=provider.name,
                        error=error,
                        latency_seconds=attempt_latency,
                    )
                execution.failed_attempts += 1
                raise

        if last_recoverable_error is not None:
            if recoverable_failures == 1:
                raise last_recoverable_error
            raise LLMProviderError("All task providers failed") from None

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

    @asynccontextmanager
    async def _provider_slot(
        self,
        provider_name: str,
        remaining: float | None,
    ) -> AsyncIterator[None]:
        if self.concurrency_manager is None:
            yield
            return
        async with self.concurrency_manager.slot(
            provider_name,
            timeout_seconds=remaining,
        ):
            yield

    @asynccontextmanager
    async def _observe_attempt(
        self,
        provider: str,
        model: str | None,
        task: TaskType,
        *,
        streaming: bool,
        fallback_depth: int,
    ) -> AsyncIterator[None]:
        started = self._clock()
        fields: dict[str, Any] = dict(
            provider=provider,
            model=model,
            task=task.value,
            streaming=streaming,
            fallback_depth=fallback_depth,
        )
        log_event(logger, "provider_attempt_started", **fields)
        try:
            yield
        except BaseException as error:
            category = (
                "budget_timeout"
                if isinstance(error, _RequestBudgetExceeded)
                else error_category(error)
            )
            if streaming and self.telemetry is not None:
                await self.telemetry.record_provider_stream_result(
                    provider,
                    success=False,
                    latency_seconds=self._clock() - started,
                    error_category=category,
                )
            log_event(
                logger,
                "provider_attempt_failed",
                level=logging.WARNING,
                **fields,
                duration_ms=max(0.0, self._clock() - started) * 1000,
                error_category=category,
                exception_type=type(error).__name__,
            )
            raise
        else:
            if streaming and self.telemetry is not None:
                await self.telemetry.record_provider_stream_result(
                    provider,
                    success=True,
                    latency_seconds=self._clock() - started,
                )
            log_event(
                logger,
                "provider_attempt_completed",
                **fields,
                duration_ms=max(0.0, self._clock() - started) * 1000,
            )
