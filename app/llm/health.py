import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import logging
import time

from app.llm.exceptions import (
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)


logger = logging.getLogger(__name__)

RECOVERABLE_HEALTH_ERRORS = (
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)


@dataclass(slots=True)
class ProviderHealthState:
    consecutive_failures: int = 0
    circuit_open_until: float | None = None
    last_failure_type: str | None = None
    last_failure_time: float | None = None


class ProviderHealthManager:
    def __init__(
        self,
        *,
        enabled: bool = True,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")

        self.enabled = enabled
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._states: dict[str, ProviderHealthState] = {}
        self._lock = asyncio.Lock()

    async def is_available(self, provider_name: str) -> bool:
        if not self.enabled:
            return True

        async with self._lock:
            state = self._states.get(provider_name)
            if state is None or state.circuit_open_until is None:
                return True

            if self._clock() >= state.circuit_open_until:
                state.circuit_open_until = None
                logger.info(
                    "Provider %s circuit eligible again",
                    provider_name,
                )
                return True
            return False

    async def record_failure(
        self,
        provider_name: str,
        error: Exception,
    ) -> None:
        if not self.enabled or not isinstance(error, RECOVERABLE_HEALTH_ERRORS):
            return

        async with self._lock:
            now = self._clock()
            state = self._states.setdefault(
                provider_name,
                ProviderHealthState(),
            )
            state.consecutive_failures += 1
            state.last_failure_type = type(error).__name__
            state.last_failure_time = now

            circuit_is_open = (
                state.circuit_open_until is not None
                and now < state.circuit_open_until
            )
            if (
                state.consecutive_failures >= self.failure_threshold
                and not circuit_is_open
            ):
                state.circuit_open_until = now + self.cooldown_seconds
                logger.warning(
                    "Provider %s circuit opened",
                    provider_name,
                )

    async def record_success(self, provider_name: str) -> None:
        if not self.enabled:
            return

        async with self._lock:
            state = self._states.get(provider_name)
            if state is None:
                return
            had_failures = (
                state.consecutive_failures > 0
                or state.circuit_open_until is not None
            )
            state.consecutive_failures = 0
            state.circuit_open_until = None
            state.last_failure_type = None
            state.last_failure_time = None
            if had_failures:
                logger.info(
                    "Provider %s health reset after success",
                    provider_name,
                )

    async def snapshot(self) -> dict[str, dict[str, int | float | str | bool | None]]:
        async with self._lock:
            now = self._clock()
            return {
                name: {
                    "consecutive_failures": state.consecutive_failures,
                    "circuit_open": (
                        state.circuit_open_until is not None
                        and now < state.circuit_open_until
                    ),
                    "circuit_open_until": state.circuit_open_until,
                    "last_failure_type": state.last_failure_type,
                    "last_failure_time": state.last_failure_time,
                }
                for name, state in self._states.items()
            }
