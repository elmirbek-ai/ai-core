from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from app.api.chat import get_llm_service
from app.core.auth import AuthContext
from app.core.config import Settings
from app.core.rate_limit import (
    APIRateLimiter,
    APIRateLimitExceeded,
    enforce_stream_rate_limit,
)
from app.llm.streaming import StreamEvent
from app.llm.task import TaskType
from app.main import app


_CLIENT_KEY = "rate-limit-test-credential"
_PAYLOAD = {"messages": [{"role": "user", "content": "Hello"}]}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class SpyService:
    chat_calls: int = 0
    stream_calls: int = 0

    async def chat(
        self,
        messages: list[object],
        task: TaskType = TaskType.GENERAL,
    ) -> dict[str, str]:
        self.chat_calls += 1
        return {"provider": "groq", "model": "mock/model", "content": "ok"}

    async def stream_chat(
        self,
        messages: list[object],
        task: TaskType = TaskType.GENERAL,
    ) -> AsyncIterator[StreamEvent]:
        self.stream_calls += 1
        yield StreamEvent("meta", {"provider": "groq", "model": "mock/model"})
        yield StreamEvent("delta", {"content": "ok"})
        yield StreamEvent("done", {})


@pytest.fixture
def api_client() -> Iterator[tuple[TestClient, SpyService]]:
    service = SpyService()
    app.dependency_overrides[get_llm_service] = lambda: service
    try:
        with TestClient(app) as client:
            app.state.ai_core_auth_enabled = True
            app.state.ai_core_api_key = SecretStr(_CLIENT_KEY)
            yield client, service
    finally:
        app.dependency_overrides.clear()


def auth_headers(key: str = _CLIENT_KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_initial_bucket_is_full() -> None:
    limiter = APIRateLimiter(burst_size=3, clock=FakeClock())

    snapshot = asyncio.run(limiter.snapshot())

    assert snapshot["available_tokens"] == 3.0
    assert snapshot["capacity"] == 3


def test_request_consumes_token_and_burst_is_enforced() -> None:
    limiter = APIRateLimiter(burst_size=2, clock=FakeClock())

    async def exercise() -> dict[str, bool | float | int]:
        await limiter.acquire_request()
        await limiter.acquire_request()
        with pytest.raises(APIRateLimitExceeded):
            await limiter.acquire_request()
        return await limiter.snapshot()

    snapshot = asyncio.run(exercise())

    assert snapshot["available_tokens"] == 0.0
    assert snapshot["allowed"] == 2
    assert snapshot["rate_limited"] == 1


def test_fake_clock_partial_refill_and_retry_after() -> None:
    clock = FakeClock()
    limiter = APIRateLimiter(
        requests_per_minute=60,
        burst_size=1,
        clock=clock,
    )

    async def exercise() -> tuple[float, dict[str, bool | float | int]]:
        await limiter.acquire_request()
        clock.advance(0.25)
        with pytest.raises(APIRateLimitExceeded) as caught:
            await limiter.acquire_request()
        clock.advance(0.75)
        await limiter.acquire_request()
        return caught.value.retry_after_seconds, await limiter.snapshot()

    retry_after, snapshot = asyncio.run(exercise())

    assert retry_after == pytest.approx(0.75)
    assert snapshot["allowed"] == 2


def test_refill_never_exceeds_capacity() -> None:
    clock = FakeClock()
    limiter = APIRateLimiter(burst_size=2, clock=clock)

    async def exercise() -> dict[str, bool | float | int]:
        await limiter.acquire_request()
        clock.advance(10_000)
        return await limiter.snapshot()

    snapshot = asyncio.run(exercise())

    assert snapshot["available_tokens"] == 2.0


def test_disabled_limiter_bypasses_tokens_and_stream_limit() -> None:
    limiter = APIRateLimiter(enabled=False, burst_size=1, max_streams=1)

    async def exercise() -> dict[str, bool | float | int]:
        for _ in range(100):
            await limiter.acquire_request()
        async with limiter.stream_slot():
            async with limiter.stream_slot():
                return await limiter.snapshot()

    snapshot = asyncio.run(exercise())

    assert snapshot["allowed"] == 0
    assert snapshot["rate_limited"] == 0
    assert snapshot["active_streams"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ai_core_requests_per_minute", 0),
        ("ai_core_burst_size", 0),
        ("ai_core_max_streams", 0),
    ],
)
def test_invalid_rate_limit_config_is_rejected(field: str, value: int) -> None:
    values = {
        "_env_file": None,
        "groq_api_key": "test-provider-key",
        "ai_core_auth_enabled": False,
        field: value,
    }

    with pytest.raises(ValidationError):
        Settings(**values)


def test_one_hundred_concurrent_requests_respect_burst() -> None:
    limiter = APIRateLimiter(
        requests_per_minute=1,
        burst_size=10,
        clock=FakeClock(),
    )

    async def exercise() -> tuple[int, int, dict[str, bool | float | int]]:
        async def attempt() -> bool:
            try:
                await limiter.acquire_request()
                return True
            except APIRateLimitExceeded:
                return False

        results = await asyncio.gather(*(attempt() for _ in range(100)))
        snapshot = await limiter.snapshot()
        return sum(results), len(results) - sum(results), snapshot

    allowed, rejected, snapshot = asyncio.run(exercise())

    assert allowed == 10
    assert rejected == 90
    assert snapshot["available_tokens"] == 0.0
    assert snapshot["allowed"] == 10
    assert snapshot["rate_limited"] == 90


def test_stream_slot_released_on_success_error_and_cancellation() -> None:
    limiter = APIRateLimiter(max_streams=1)

    async def exercise() -> list[int]:
        observed: list[int] = []
        async with limiter.stream_slot():
            observed.append(int((await limiter.snapshot())["active_streams"]))
        observed.append(int((await limiter.snapshot())["active_streams"]))

        with pytest.raises(RuntimeError):
            async with limiter.stream_slot():
                raise RuntimeError("stream failed")
        observed.append(int((await limiter.snapshot())["active_streams"]))

        entered = asyncio.Event()

        async def hold() -> None:
            async with limiter.stream_slot():
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(hold())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        observed.append(int((await limiter.snapshot())["active_streams"]))
        return observed

    assert asyncio.run(exercise()) == [1, 0, 0, 0]


def test_stream_dependency_close_releases_slot() -> None:
    limiter = APIRateLimiter(burst_size=1, max_streams=1)

    async def exercise() -> tuple[int, int]:
        dependency = enforce_stream_rate_limit(
            AuthContext(client_id="default"),
            limiter,
        )
        await anext(dependency)
        during = int((await limiter.snapshot())["active_streams"])
        await dependency.aclose()
        after = int((await limiter.snapshot())["active_streams"])
        return during, after

    assert asyncio.run(exercise()) == (1, 0)


@pytest.mark.parametrize("headers", [{}, auth_headers("wrong-credential")])
def test_unauthorized_request_does_not_mutate_limiter(
    api_client: tuple[TestClient, SpyService],
    headers: dict[str, str],
) -> None:
    client, service = api_client
    before = service.chat_calls
    limiter = APIRateLimiter(burst_size=1, clock=FakeClock())
    app.state.api_rate_limiter = limiter

    response = client.post(
        "/v1/chat",
        json=_PAYLOAD,
        headers=headers,
    )
    snapshot = asyncio.run(limiter.snapshot())

    assert response.status_code == 401
    assert snapshot["allowed"] == 0
    assert snapshot["available_tokens"] == 1.0
    assert service.chat_calls == before


def test_chat_allowed_then_rate_limited_before_service(
    api_client: tuple[TestClient, SpyService],
) -> None:
    client, service = api_client
    limiter = APIRateLimiter(
        requests_per_minute=1,
        burst_size=1,
        clock=FakeClock(),
    )
    app.state.api_rate_limiter = limiter

    allowed = client.post("/v1/chat", json=_PAYLOAD, headers=auth_headers())
    rejected = client.post("/v1/chat", json=_PAYLOAD, headers=auth_headers())

    assert allowed.status_code == 200
    assert rejected.status_code == 429
    assert rejected.json() == {"detail": "Too Many Requests"}
    assert rejected.headers["content-type"].startswith("application/json")
    assert int(rejected.headers["retry-after"]) >= 1
    assert service.chat_calls == 1


def test_fake_clock_refill_allows_next_api_request(
    api_client: tuple[TestClient, SpyService],
) -> None:
    client, service = api_client
    before = service.chat_calls
    clock = FakeClock()
    limiter = APIRateLimiter(
        requests_per_minute=60,
        burst_size=1,
        clock=clock,
    )
    app.state.api_rate_limiter = limiter

    first = client.post("/v1/chat", json=_PAYLOAD, headers=auth_headers())
    exhausted = client.post("/v1/chat", json=_PAYLOAD, headers=auth_headers())
    clock.advance(1.0)
    refilled = client.post("/v1/chat", json=_PAYLOAD, headers=auth_headers())

    assert [first.status_code, exhausted.status_code, refilled.status_code] == [
        200,
        429,
        200,
    ]
    assert service.chat_calls == before + 2


def test_stream_token_exhaustion_returns_http_429_before_sse(
    api_client: tuple[TestClient, SpyService],
) -> None:
    client, service = api_client
    before = service.stream_calls
    limiter = APIRateLimiter(burst_size=1, clock=FakeClock())
    asyncio.run(limiter.acquire_request())
    app.state.api_rate_limiter = limiter

    response = client.post(
        "/v1/chat/stream",
        json=_PAYLOAD,
        headers=auth_headers(),
    )

    assert response.status_code == 429
    assert not response.headers["content-type"].startswith("text/event-stream")
    assert "event: error" not in response.text
    assert service.stream_calls == before


def test_stream_provider_failure_releases_api_slot(
    api_client: tuple[TestClient, SpyService],
) -> None:
    client, _ = api_client
    limiter = APIRateLimiter(burst_size=1, max_streams=1)

    class FailingStreamService:
        async def stream_chat(self, messages, task=TaskType.GENERAL):
            raise RuntimeError("sanitized by endpoint")
            yield

    app.state.api_rate_limiter = limiter
    app.dependency_overrides[get_llm_service] = lambda: FailingStreamService()
    try:
        response = client.post(
            "/v1/chat/stream",
            json=_PAYLOAD,
            headers=auth_headers(),
        )
    finally:
        app.dependency_overrides[get_llm_service] = lambda: api_client[1]

    snapshot = asyncio.run(limiter.snapshot())
    assert response.status_code == 200
    assert "event: error" in response.text
    assert snapshot["active_streams"] == 0


def test_ten_concurrent_streams_respect_max_three() -> None:
    async def exercise() -> tuple[list[int], int, dict[str, bool | float | int]]:
        limiter = APIRateLimiter(burst_size=10, max_streams=3)
        release = asyncio.Event()
        three_active = asyncio.Event()

        class HoldingService:
            def __init__(self) -> None:
                self.calls = 0
                self.active = 0
                self.max_active = 0

            async def stream_chat(self, messages, task=TaskType.GENERAL):
                self.calls += 1
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                if self.active == 3:
                    three_active.set()
                try:
                    await release.wait()
                    yield StreamEvent(
                        "meta",
                        {"provider": "groq", "model": "mock/model"},
                    )
                    yield StreamEvent("done", {})
                finally:
                    self.active -= 1

        service = HoldingService()
        app.state.ai_core_auth_enabled = True
        app.state.ai_core_api_key = SecretStr(_CLIENT_KEY)
        app.state.api_rate_limiter = limiter
        app.dependency_overrides[get_llm_service] = lambda: service
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                tasks = [
                    asyncio.create_task(
                        client.post(
                            "/v1/chat/stream",
                            json=_PAYLOAD,
                            headers=auth_headers(),
                        )
                    )
                    for _ in range(10)
                ]
                await asyncio.wait_for(three_active.wait(), timeout=1.0)
                await asyncio.sleep(0)
                during = await limiter.snapshot()
                release.set()
                responses = await asyncio.gather(*tasks)
        finally:
            app.dependency_overrides.clear()

        return [response.status_code for response in responses], service.max_active, during

    statuses, max_active, during = asyncio.run(exercise())

    assert statuses.count(200) == 3
    assert statuses.count(429) == 7
    assert max_active == 3
    assert during["active_streams"] == 3
    assert asyncio.run(app.state.api_rate_limiter.snapshot())["active_streams"] == 0
