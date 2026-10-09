import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.task import TaskType


@dataclass(slots=True)
class ProviderMetrics:
    attempts: int = 0
    successes: int = 0
    recoverable_failures: int = 0
    authentication_failures: int = 0
    provider_failures: int = 0
    timeout_failures: int = 0
    rate_limit_failures: int = 0
    upstream_failures: int = 0
    circuit_skips: int = 0
    total_latency_seconds: float = 0.0
    last_latency_seconds: float = 0.0
    last_success_time: float | None = None
    last_failure_time: float | None = None
    streaming_attempts: int = 0
    streaming_successes: int = 0
    streaming_failures: int = 0
    streaming_cancellations: int = 0
    streaming_total_latency_seconds: float = 0.0
    streaming_last_latency_seconds: float = 0.0
    streaming_failures_by_category: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class StreamingRequestMetrics:
    total: int = 0
    successes: int = 0
    failures: int = 0
    cancellations: int = 0
    total_duration_seconds: float = 0.0
    total_time_to_first_token_seconds: float = 0.0
    first_token_count: int = 0
    last_duration_seconds: float = 0.0
    last_time_to_first_token_seconds: float | None = None
    last_selected_provider: str | None = None
    last_task: str | None = None


@dataclass(slots=True)
class TaskRequestMetrics:
    total: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_seconds: float = 0.0


@dataclass(slots=True)
class RequestMetrics:
    total: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_seconds: float = 0.0
    last_latency_seconds: float = 0.0
    total_fallback_depth: int = 0
    last_fallback_depth: int = 0
    last_selected_provider: str | None = None
    last_task: str | None = None
    last_error_category: str | None = None
    by_task: dict[str, TaskRequestMetrics] = field(default_factory=dict)


class LLMTelemetry:
    def __init__(
        self,
        *,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = enabled
        self._clock = clock
        self._providers: dict[str, ProviderMetrics] = {}
        self._requests = RequestMetrics()
        self._streaming = StreamingRequestMetrics()
        self._lock = asyncio.Lock()

    async def record_provider_success(
        self,
        provider_name: str,
        latency_seconds: float,
    ) -> None:
        if not self.enabled:
            return
        async with self._lock:
            metrics = self._providers.setdefault(
                provider_name,
                ProviderMetrics(),
            )
            metrics.attempts += 1
            metrics.successes += 1
            self._record_provider_latency(metrics, latency_seconds)
            metrics.last_success_time = self._clock()

    async def record_provider_failure(
        self,
        provider_name: str,
        error: Exception,
        latency_seconds: float,
    ) -> None:
        if not self.enabled:
            return
        async with self._lock:
            metrics = self._providers.setdefault(
                provider_name,
                ProviderMetrics(),
            )
            metrics.attempts += 1
            self._record_provider_latency(metrics, latency_seconds)
            metrics.last_failure_time = self._clock()

            if isinstance(error, LLMAuthenticationError):
                metrics.authentication_failures += 1
            elif isinstance(error, LLMTimeoutError):
                metrics.recoverable_failures += 1
                metrics.timeout_failures += 1
            elif isinstance(error, LLMRateLimitError):
                metrics.recoverable_failures += 1
                metrics.rate_limit_failures += 1
            elif isinstance(error, LLMUpstreamError):
                metrics.recoverable_failures += 1
                metrics.upstream_failures += 1
            elif isinstance(error, LLMProviderError):
                metrics.provider_failures += 1
            else:
                metrics.provider_failures += 1

    async def record_circuit_skip(self, provider_name: str) -> None:
        if not self.enabled:
            return
        async with self._lock:
            metrics = self._providers.setdefault(
                provider_name,
                ProviderMetrics(),
            )
            metrics.circuit_skips += 1

    async def record_provider_stream_result(
        self,
        provider_name: str,
        *,
        success: bool,
        latency_seconds: float = 0.0,
        error_category: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        async with self._lock:
            metrics = self._providers.setdefault(
                provider_name,
                ProviderMetrics(),
            )
            metrics.streaming_attempts += 1
            latency = max(0.0, latency_seconds)
            metrics.streaming_total_latency_seconds += latency
            metrics.streaming_last_latency_seconds = latency
            if success:
                metrics.streaming_successes += 1
            else:
                metrics.streaming_failures += 1
                category = (
                    error_category
                    if error_category
                    in {
                        "authentication",
                        "rate_limit",
                        "timeout",
                        "upstream",
                        "provider",
                        "budget_timeout",
                        "cancelled",
                    }
                    else "provider"
                )
                metrics.streaming_failures_by_category[category] = (
                    metrics.streaming_failures_by_category.get(category, 0) + 1
                )
                if category == "cancelled":
                    metrics.streaming_cancellations += 1

    async def record_stream_request(
        self,
        *,
        task: TaskType,
        selected_provider: str | None,
        success: bool,
        duration_seconds: float,
        time_to_first_token_seconds: float | None,
        cancelled: bool = False,
    ) -> None:
        if not self.enabled:
            return
        async with self._lock:
            duration = max(0.0, duration_seconds)
            metrics = self._streaming
            metrics.total += 1
            metrics.total_duration_seconds += duration
            metrics.last_duration_seconds = duration
            metrics.last_selected_provider = selected_provider
            metrics.last_task = task.value
            if success:
                metrics.successes += 1
            else:
                metrics.failures += 1
            if cancelled:
                metrics.cancellations += 1
            metrics.last_time_to_first_token_seconds = None
            if time_to_first_token_seconds is not None:
                first_token = max(0.0, time_to_first_token_seconds)
                metrics.total_time_to_first_token_seconds += first_token
                metrics.first_token_count += 1
                metrics.last_time_to_first_token_seconds = first_token

    async def record_request_success(
        self,
        task: TaskType,
        selected_provider: str,
        latency_seconds: float,
        fallback_depth: int,
    ) -> None:
        await self._record_request(
            task=task,
            latency_seconds=latency_seconds,
            fallback_depth=fallback_depth,
            selected_provider=selected_provider,
            error=None,
        )

    async def record_request_failure(
        self,
        task: TaskType,
        error: Exception,
        latency_seconds: float,
        fallback_depth: int,
    ) -> None:
        await self._record_request(
            task=task,
            latency_seconds=latency_seconds,
            fallback_depth=fallback_depth,
            selected_provider=None,
            error=error,
        )

    async def snapshot(self) -> dict:
        async with self._lock:
            providers = {
                name: {
                    "attempts": metrics.attempts,
                    "successes": metrics.successes,
                    "recoverable_failures": metrics.recoverable_failures,
                    "authentication_failures": (metrics.authentication_failures),
                    "provider_failures": metrics.provider_failures,
                    "timeout_failures": metrics.timeout_failures,
                    "rate_limit_failures": metrics.rate_limit_failures,
                    "upstream_failures": metrics.upstream_failures,
                    "circuit_skips": metrics.circuit_skips,
                    "total_latency_seconds": metrics.total_latency_seconds,
                    "last_latency_seconds": metrics.last_latency_seconds,
                    "average_latency_seconds": self._average(
                        metrics.total_latency_seconds,
                        metrics.attempts,
                    ),
                    "last_success_time": metrics.last_success_time,
                    "last_failure_time": metrics.last_failure_time,
                    "streaming_attempts": metrics.streaming_attempts,
                    "streaming_successes": metrics.streaming_successes,
                    "streaming_failures": metrics.streaming_failures,
                    "streaming_cancellations": metrics.streaming_cancellations,
                    "streaming_total_latency_seconds": (
                        metrics.streaming_total_latency_seconds
                    ),
                    "streaming_last_latency_seconds": (
                        metrics.streaming_last_latency_seconds
                    ),
                    "streaming_average_latency_seconds": self._average(
                        metrics.streaming_total_latency_seconds,
                        metrics.streaming_attempts,
                    ),
                    "streaming_failures_by_category": dict(
                        metrics.streaming_failures_by_category
                    ),
                }
                for name, metrics in self._providers.items()
            }
            requests = {
                "total": self._requests.total,
                "successes": self._requests.successes,
                "failures": self._requests.failures,
                "total_latency_seconds": self._requests.total_latency_seconds,
                "last_latency_seconds": self._requests.last_latency_seconds,
                "average_latency_seconds": self._average(
                    self._requests.total_latency_seconds,
                    self._requests.total,
                ),
                "total_fallback_depth": self._requests.total_fallback_depth,
                "last_fallback_depth": self._requests.last_fallback_depth,
                "average_fallback_depth": self._average(
                    float(self._requests.total_fallback_depth),
                    self._requests.total,
                ),
                "last_selected_provider": (self._requests.last_selected_provider),
                "last_task": self._requests.last_task,
                "last_error_category": self._requests.last_error_category,
                "by_task": {
                    name: {
                        "total": metrics.total,
                        "successes": metrics.successes,
                        "failures": metrics.failures,
                        "total_latency_seconds": (metrics.total_latency_seconds),
                        "average_latency_seconds": self._average(
                            metrics.total_latency_seconds,
                            metrics.total,
                        ),
                    }
                    for name, metrics in self._requests.by_task.items()
                },
            }
            streaming = {
                "total": self._streaming.total,
                "successes": self._streaming.successes,
                "failures": self._streaming.failures,
                "cancellations": self._streaming.cancellations,
                "total_duration_seconds": (self._streaming.total_duration_seconds),
                "average_duration_seconds": self._average(
                    self._streaming.total_duration_seconds,
                    self._streaming.total,
                ),
                "total_time_to_first_token_seconds": (
                    self._streaming.total_time_to_first_token_seconds
                ),
                "average_time_to_first_token_seconds": self._average(
                    self._streaming.total_time_to_first_token_seconds,
                    self._streaming.first_token_count,
                ),
                "last_duration_seconds": (self._streaming.last_duration_seconds),
                "last_time_to_first_token_seconds": (
                    self._streaming.last_time_to_first_token_seconds
                ),
                "last_selected_provider": (self._streaming.last_selected_provider),
                "last_task": self._streaming.last_task,
            }
            return {
                "providers": providers,
                "requests": requests,
                "streaming": streaming,
            }

    async def _record_request(
        self,
        *,
        task: TaskType,
        latency_seconds: float,
        fallback_depth: int,
        selected_provider: str | None,
        error: Exception | None,
    ) -> None:
        if not self.enabled:
            return
        async with self._lock:
            latency = max(0.0, latency_seconds)
            depth = max(0, fallback_depth)
            self._requests.total += 1
            self._requests.total_latency_seconds += latency
            self._requests.last_latency_seconds = latency
            self._requests.total_fallback_depth += depth
            self._requests.last_fallback_depth = depth
            self._requests.last_selected_provider = selected_provider
            self._requests.last_task = task.value
            self._requests.last_error_category = (
                self._error_category(error) if error is not None else None
            )

            task_metrics = self._requests.by_task.setdefault(
                task.value,
                TaskRequestMetrics(),
            )
            task_metrics.total += 1
            task_metrics.total_latency_seconds += latency
            if error is None:
                self._requests.successes += 1
                task_metrics.successes += 1
            else:
                self._requests.failures += 1
                task_metrics.failures += 1

    @staticmethod
    def _record_provider_latency(
        metrics: ProviderMetrics,
        latency_seconds: float,
    ) -> None:
        latency = max(0.0, latency_seconds)
        metrics.total_latency_seconds += latency
        metrics.last_latency_seconds = latency

    @staticmethod
    def _average(total: float, count: int) -> float:
        return total / count if count else 0.0

    @staticmethod
    def _error_category(error: Exception) -> str:
        if isinstance(error, LLMAuthenticationError):
            return "authentication"
        if isinstance(error, LLMTimeoutError):
            return "timeout"
        if isinstance(error, LLMRateLimitError):
            return "rate_limit"
        if isinstance(error, LLMUpstreamError):
            return "upstream"
        if isinstance(error, LLMProviderError):
            return "provider"
        return "unknown"
