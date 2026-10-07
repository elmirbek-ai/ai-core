import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
import time

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
                    "authentication_failures": (
                        metrics.authentication_failures
                    ),
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
                "last_selected_provider": (
                    self._requests.last_selected_provider
                ),
                "last_task": self._requests.last_task,
                "last_error_category": self._requests.last_error_category,
                "by_task": {
                    name: {
                        "total": metrics.total,
                        "successes": metrics.successes,
                        "failures": metrics.failures,
                        "total_latency_seconds": (
                            metrics.total_latency_seconds
                        ),
                        "average_latency_seconds": self._average(
                            metrics.total_latency_seconds,
                            metrics.total,
                        ),
                    }
                    for name, metrics in self._requests.by_task.items()
                },
            }
            return {"providers": providers, "requests": requests}

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
