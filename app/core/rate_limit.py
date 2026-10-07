from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, NoReturn

from fastapi import Depends, HTTPException, Request, status

from app.core.auth import AuthContext, verify_api_key


class APIRateLimitExceeded(Exception):
    def __init__(self, retry_after_seconds: float) -> None:
        super().__init__("AI Core inbound rate limit exceeded")
        self.retry_after_seconds = retry_after_seconds


class APIRateLimiter:
    def __init__(
        self,
        *,
        enabled: bool = True,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        max_streams: int = 4,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1")
        if burst_size < 1:
            raise ValueError("burst_size must be at least 1")
        if max_streams < 1:
            raise ValueError("max_streams must be at least 1")

        self.enabled = enabled
        self.requests_per_minute = requests_per_minute
        self.capacity = burst_size
        self.max_streams = max_streams
        self._clock = clock
        self._refill_per_second = requests_per_minute / 60.0
        self._tokens = float(burst_size)
        self._last_refill = clock()
        self._allowed = 0
        self._rate_limited = 0
        self._active_streams = 0
        self._lock = asyncio.Lock()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_refill)
        if elapsed:
            self._tokens = min(
                float(self.capacity),
                self._tokens + elapsed * self._refill_per_second,
            )
            self._last_refill = now

    async def acquire_request(self, client_id: str = "default") -> None:
        del client_id  # Reserved for future sanitized multi-client buckets.
        if not self.enabled:
            return

        async with self._lock:
            self._refill(self._clock())
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self._allowed += 1
                return

            self._rate_limited += 1
            retry_after = (1.0 - self._tokens) / self._refill_per_second
            raise APIRateLimitExceeded(retry_after)

    @asynccontextmanager
    async def stream_slot(self) -> AsyncIterator[None]:
        if not self.enabled:
            yield
            return

        async with self._lock:
            if self._active_streams >= self.max_streams:
                self._rate_limited += 1
                raise APIRateLimitExceeded(1.0)
            self._active_streams += 1

        try:
            yield
        finally:
            async with self._lock:
                self._active_streams -= 1

    async def snapshot(self) -> dict[str, bool | float | int]:
        async with self._lock:
            if self.enabled:
                self._refill(self._clock())
            return {
                "enabled": self.enabled,
                "available_tokens": self._tokens,
                "capacity": self.capacity,
                "allowed": self._allowed,
                "rate_limited": self._rate_limited,
                "active_streams": self._active_streams,
                "max_streams": self.max_streams,
            }


def get_api_rate_limiter(request: Request) -> APIRateLimiter:
    return request.app.state.api_rate_limiter


def _raise_http_rate_limit(error: APIRateLimitExceeded) -> NoReturn:
    retry_after = max(1, math.ceil(error.retry_after_seconds))
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Too Many Requests",
        headers={"Retry-After": str(retry_after)},
    ) from None


async def enforce_api_rate_limit(
    auth: Annotated[AuthContext, Depends(verify_api_key)],
    limiter: Annotated[APIRateLimiter, Depends(get_api_rate_limiter)],
) -> None:
    try:
        await limiter.acquire_request(auth.client_id)
    except APIRateLimitExceeded as error:
        _raise_http_rate_limit(error)


async def enforce_stream_rate_limit(
    auth: Annotated[AuthContext, Depends(verify_api_key)],
    limiter: Annotated[APIRateLimiter, Depends(get_api_rate_limiter)],
) -> AsyncIterator[None]:
    try:
        await limiter.acquire_request(auth.client_id)
        async with limiter.stream_slot():
            yield
    except APIRateLimitExceeded as error:
        _raise_http_rate_limit(error)
