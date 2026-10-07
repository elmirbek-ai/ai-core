import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass


class ProviderConcurrencyTimeout(Exception):
    pass


@dataclass(slots=True)
class _ProviderLimiter:
    limit: int
    semaphore: asyncio.Semaphore
    in_flight: int = 0


class ProviderConcurrencyManager:
    def __init__(
        self,
        *,
        enabled: bool = True,
        default_limit: int = 4,
        limits: dict[str, int] | None = None,
    ) -> None:
        if default_limit < 1:
            raise ValueError("default_limit must be at least 1")
        configured_limits = dict(limits or {})
        if any(limit < 1 for limit in configured_limits.values()):
            raise ValueError("provider concurrency limits must be at least 1")

        self.enabled = enabled
        self.default_limit = default_limit
        self._configured_limits = configured_limits
        self._limiters: dict[str, _ProviderLimiter] = {
            name: self._new_limiter(limit)
            for name, limit in configured_limits.items()
        }
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def slot(
        self,
        provider_name: str,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[None]:
        if not self.enabled:
            yield
            return
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ProviderConcurrencyTimeout from None

        limiter = await self._get_limiter(provider_name)
        try:
            if timeout_seconds is None:
                await limiter.semaphore.acquire()
            else:
                async with asyncio.timeout(timeout_seconds):
                    await limiter.semaphore.acquire()
        except TimeoutError:
            raise ProviderConcurrencyTimeout from None

        limiter.in_flight += 1
        try:
            yield
        finally:
            limiter.in_flight -= 1
            limiter.semaphore.release()

    async def snapshot(self) -> dict[str, dict[str, int]]:
        async with self._lock:
            return {
                name: {
                    "limit": limiter.limit,
                    "in_flight": limiter.in_flight,
                    "available": limiter.limit - limiter.in_flight,
                }
                for name, limiter in self._limiters.items()
            }

    async def _get_limiter(self, provider_name: str) -> _ProviderLimiter:
        limiter = self._limiters.get(provider_name)
        if limiter is not None:
            return limiter
        async with self._lock:
            return self._limiters.setdefault(
                provider_name,
                self._new_limiter(
                    self._configured_limits.get(
                        provider_name,
                        self.default_limit,
                    )
                ),
            )

    @staticmethod
    def _new_limiter(limit: int) -> _ProviderLimiter:
        return _ProviderLimiter(
            limit=limit,
            semaphore=asyncio.Semaphore(limit),
        )
