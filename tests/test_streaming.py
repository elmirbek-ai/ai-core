import asyncio
import json
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.api.chat import get_llm_service, serialize_sse
from app.llm.budget import RequestBudgetPolicy
from app.llm.capabilities import ProviderCapabilities
from app.llm.concurrency import ProviderConcurrencyManager
from app.llm.exceptions import LLMTimeoutError, LLMUpstreamError
from app.llm.health import ProviderHealthManager
from app.llm.providers.cloudflare import CloudflareProvider
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.groq import GroqProvider
from app.llm.providers.kilo import KiloProvider
from app.llm.providers.llm7 import LLM7Provider
from app.llm.providers.ollama import OllamaProvider
from app.llm.providers.openrouter import OpenRouterProvider
from app.llm.router import LLMRouter
from app.llm.streaming import ProviderStreamChunk, StreamEvent
from app.llm.task import TaskType
from app.llm.telemetry import LLMTelemetry
from app.main import app

MESSAGES = [{"role": "user", "content": "Hello"}]
StreamFactory = Callable[[], AsyncIterator[ProviderStreamChunk]]


class MockStreamingProvider:
    def __init__(
        self,
        name: str,
        factory: StreamFactory,
        *,
        streaming: bool = True,
    ) -> None:
        self.name = name
        self.factory = factory
        self.capabilities = ProviderCapabilities(streaming=streaming)
        self.stream_calls = 0
        self.active = 0

    async def stream_chat(self, messages, model=None):
        self.stream_calls += 1
        self.active += 1
        try:
            async for chunk in self.factory():
                yield chunk
        finally:
            self.active -= 1


def chunk_stream(
    provider: str,
    chunks: list[str],
    *,
    error: Exception | None = None,
) -> StreamFactory:
    async def factory() -> AsyncIterator[ProviderStreamChunk]:
        for content in chunks:
            await asyncio.sleep(0)
            yield ProviderStreamChunk(provider, f"mock/{provider}", content)
        if error is not None:
            raise error

    return factory


def error_stream(error: Exception) -> StreamFactory:
    async def factory() -> AsyncIterator[ProviderStreamChunk]:
        raise error
        yield

    return factory


def hanging_stream(
    provider: str,
    *,
    first_chunk: str | None = None,
    entered: asyncio.Event | None = None,
    release: asyncio.Event | None = None,
) -> StreamFactory:
    async def factory() -> AsyncIterator[ProviderStreamChunk]:
        if first_chunk is not None:
            yield ProviderStreamChunk(
                provider,
                f"mock/{provider}",
                first_chunk,
            )
        if entered is not None:
            entered.set()
        if release is None:
            await asyncio.Event().wait()
        else:
            await release.wait()

    return factory


async def collect(router: LLMRouter) -> list[StreamEvent]:
    return [event async for event in router.stream_chat(MESSAGES)]


def short_budget(seconds: float = 0.02) -> RequestBudgetPolicy:
    return RequestBudgetPolicy(
        standard_seconds=seconds,
        reasoning_seconds=seconds,
        code_seconds=seconds,
        long_context_seconds=seconds,
        multimodal_seconds=seconds,
    )


class FakeStreamingService:
    async def stream_chat(self, messages, task=TaskType.GENERAL):
        yield StreamEvent(
            "meta",
            {"provider": "groq", "model": "openai/gpt-oss-20b"},
        )
        yield StreamEvent("delta", {"content": "Hello\n"})
        yield StreamEvent("delta", {"content": '"world"'})
        yield StreamEvent("done", {})


def test_sse_endpoint_contract_and_json_escaping() -> None:
    app.dependency_overrides[get_llm_service] = lambda: FakeStreamingService()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/stream",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: meta\n" in response.text
    assert response.text.count("event: delta\n") == 2
    assert "event: done\ndata: {}\n\n" in response.text
    assert 'data: {"content":"Hello\\n"}' in response.text
    assert 'data: {"content":"\\"world\\""}' in response.text


def test_sse_serializer_never_uses_done_sentinel() -> None:
    serialized = serialize_sse("done", {})

    assert serialized == "event: done\ndata: {}\n\n"
    assert "[DONE]" not in serialized


def test_primary_streaming_success_emits_meta_deltas_done() -> None:
    primary = MockStreamingProvider("groq", chunk_stream("groq", ["Hel", "lo"]))
    router = LLMRouter(primary)

    events = asyncio.run(collect(router))

    assert [event.event for event in events] == ["meta", "delta", "delta", "done"]
    assert events[0].data == {"provider": "groq", "model": "mock/groq"}
    assert [event.data.get("content") for event in events[1:3]] == ["Hel", "lo"]


def test_pre_token_recoverable_failure_falls_back() -> None:
    primary = MockStreamingProvider(
        "groq",
        error_stream(LLMTimeoutError("private")),
    )
    fallback = MockStreamingProvider(
        "openrouter",
        chunk_stream("openrouter", ["ok"]),
    )
    router = LLMRouter(primary, fallback)

    events = asyncio.run(collect(router))

    assert events[0].data["provider"] == "openrouter"
    assert primary.stream_calls == 1
    assert fallback.stream_calls == 1


def test_non_streaming_capability_is_skipped_without_attempt() -> None:
    primary = MockStreamingProvider(
        "groq",
        chunk_stream("groq", ["must not run"]),
        streaming=False,
    )
    fallback = MockStreamingProvider(
        "openrouter",
        chunk_stream("openrouter", ["ok"]),
    )
    telemetry = LLMTelemetry()
    router = LLMRouter(primary, fallback, telemetry=telemetry)

    async def exercise() -> tuple[list[StreamEvent], dict]:
        return await collect(router), await telemetry.snapshot()

    events, metrics = asyncio.run(exercise())

    assert events[0].data["provider"] == "openrouter"
    assert primary.stream_calls == 0
    assert "groq" not in metrics["providers"]


def test_post_token_failure_never_falls_back() -> None:
    primary = MockStreamingProvider(
        "groq",
        chunk_stream(
            "groq",
            ["partial"],
            error=LLMUpstreamError("private"),
        ),
    )
    fallback = MockStreamingProvider(
        "openrouter",
        chunk_stream("openrouter", ["must not run"]),
    )
    router = LLMRouter(primary, fallback)

    async def exercise() -> tuple[list[StreamEvent], type[Exception]]:
        events = []
        with pytest.raises(LLMUpstreamError) as captured:
            async for event in router.stream_chat(MESSAGES):
                events.append(event)
        return events, type(captured.value)

    events, error_type = asyncio.run(exercise())

    assert error_type is LLMUpstreamError
    assert [event.event for event in events] == ["meta", "delta"]
    assert fallback.stream_calls == 0


def test_api_emits_generic_error_after_partial_stream() -> None:
    class FailingService:
        async def stream_chat(self, messages, task=TaskType.GENERAL):
            yield StreamEvent("meta", {"provider": "groq", "model": "model"})
            yield StreamEvent("delta", {"content": "partial"})
            raise LLMUpstreamError("raw private upstream body")

    app.dependency_overrides[get_llm_service] = lambda: FailingService()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/stream",
                json={"messages": [{"role": "user", "content": "secret prompt"}]},
            )
    finally:
        app.dependency_overrides.clear()

    assert "event: error" in response.text
    assert "temporarily unavailable" in response.text
    assert "raw private upstream body" not in response.text
    assert "secret prompt" not in response.text


@pytest.mark.parametrize("first_chunk", [None, "partial"])
def test_stream_deadline_before_or_after_first_token(first_chunk: str | None) -> None:
    primary = MockStreamingProvider(
        "groq",
        hanging_stream("groq", first_chunk=first_chunk),
    )
    fallback = MockStreamingProvider(
        "openrouter",
        chunk_stream("openrouter", ["must not run"]),
    )
    health = ProviderHealthManager(failure_threshold=3)
    telemetry = LLMTelemetry()
    router = LLMRouter(
        primary,
        fallback,
        health_manager=health,
        budget_policy=short_budget(),
        telemetry=telemetry,
    )

    async def exercise() -> tuple[list[StreamEvent], dict, dict]:
        events = []
        with pytest.raises(LLMTimeoutError):
            async for event in router.stream_chat(MESSAGES):
                events.append(event)
        return events, await health.snapshot(), await telemetry.snapshot()

    events, health_state, metrics = asyncio.run(exercise())

    expected = [] if first_chunk is None else ["meta", "delta"]
    assert [event.event for event in events] == expected
    assert fallback.stream_calls == 0
    assert health_state["groq"]["consecutive_failures"] == 1
    assert metrics["providers"]["groq"]["streaming_failures"] == 1
    assert metrics["streaming"]["failures"] == 1


def test_concurrency_slot_is_held_for_stream_and_released_on_success() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    primary = MockStreamingProvider(
        "groq",
        hanging_stream(
            "groq",
            first_chunk="first",
            entered=entered,
            release=release,
        ),
    )
    manager = ProviderConcurrencyManager(limits={"groq": 1})
    router = LLMRouter(primary, concurrency_manager=manager)

    async def exercise() -> tuple[dict, dict]:
        async def consume() -> None:
            async for _ in router.stream_chat(MESSAGES):
                pass

        task = asyncio.create_task(consume())
        await entered.wait()
        during = await manager.snapshot()
        release.set()
        await task
        return during, await manager.snapshot()

    during, after = asyncio.run(exercise())

    assert during["groq"]["in_flight"] == 1
    assert after["groq"]["in_flight"] == 0


def test_concurrency_slot_released_on_provider_failure() -> None:
    manager = ProviderConcurrencyManager(limits={"groq": 1})
    primary = MockStreamingProvider(
        "groq",
        error_stream(LLMUpstreamError("private")),
    )
    router = LLMRouter(primary, concurrency_manager=manager)

    async def exercise() -> dict:
        with pytest.raises(LLMUpstreamError):
            await collect(router)
        return await manager.snapshot()

    snapshot = asyncio.run(exercise())

    assert snapshot["groq"]["in_flight"] == 0
    assert snapshot["groq"]["available"] == 1


def test_slot_wait_timeout_is_not_a_provider_attempt_or_health_failure() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    primary = MockStreamingProvider(
        "groq",
        hanging_stream(
            "groq",
            first_chunk="first",
            entered=entered,
            release=release,
        ),
    )
    manager = ProviderConcurrencyManager(limits={"groq": 1})
    health = ProviderHealthManager(failure_threshold=1)
    telemetry = LLMTelemetry()
    long_router = LLMRouter(primary, concurrency_manager=manager)
    short_router = LLMRouter(
        primary,
        health_manager=health,
        budget_policy=short_budget(),
        telemetry=telemetry,
        concurrency_manager=manager,
    )

    async def exercise() -> tuple[dict, dict, dict]:
        first = asyncio.create_task(collect(long_router))
        await entered.wait()
        with pytest.raises(LLMTimeoutError):
            await collect(short_router)
        calls_after_timeout = primary.stream_calls
        health_state = await health.snapshot()
        metrics = await telemetry.snapshot()
        release.set()
        await first
        concurrency = await manager.snapshot()
        return (
            {
                "calls": calls_after_timeout,
                "in_flight": concurrency["groq"]["in_flight"],
            },
            health_state,
            metrics,
        )

    outcome, health_state, metrics = asyncio.run(exercise())

    assert outcome == {"calls": 1, "in_flight": 0}
    assert health_state == {}
    assert metrics["providers"] == {}
    assert metrics["streaming"]["failures"] == 1


def test_client_close_releases_slot_without_health_failure() -> None:
    primary = MockStreamingProvider(
        "groq",
        hanging_stream("groq", first_chunk="first"),
    )
    manager = ProviderConcurrencyManager(limits={"groq": 1})
    health = ProviderHealthManager(failure_threshold=1)
    telemetry = LLMTelemetry()
    router = LLMRouter(
        primary,
        health_manager=health,
        telemetry=telemetry,
        concurrency_manager=manager,
    )

    async def exercise() -> tuple[dict, dict, dict]:
        stream = router.stream_chat(MESSAGES)
        assert (await anext(stream)).event == "meta"
        assert (await anext(stream)).event == "delta"
        await stream.aclose()
        return (
            await manager.snapshot(),
            await health.snapshot(),
            await telemetry.snapshot(),
        )

    concurrency, health_state, metrics = asyncio.run(exercise())

    assert concurrency["groq"]["in_flight"] == 0
    assert primary.active == 0
    assert health_state == {}
    assert metrics["providers"]["groq"]["attempts"] == 0
    assert metrics["providers"]["groq"]["streaming_attempts"] == 1
    assert metrics["providers"]["groq"]["streaming_cancellations"] == 1
    assert metrics["providers"]["groq"]["streaming_failures_by_category"] == {
        "cancelled": 1
    }
    assert metrics["streaming"]["cancellations"] == 1


def test_task_cancellation_releases_slot_without_health_failure() -> None:
    entered = asyncio.Event()
    primary = MockStreamingProvider(
        "groq",
        hanging_stream("groq", entered=entered),
    )
    manager = ProviderConcurrencyManager(limits={"groq": 1})
    health = ProviderHealthManager(failure_threshold=1)
    telemetry = LLMTelemetry()
    router = LLMRouter(
        primary,
        health_manager=health,
        telemetry=telemetry,
        concurrency_manager=manager,
    )

    async def exercise() -> tuple[dict, dict, dict]:
        task = asyncio.create_task(collect(router))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return (
            await manager.snapshot(),
            await health.snapshot(),
            await telemetry.snapshot(),
        )

    concurrency, health_state, metrics = asyncio.run(exercise())

    assert concurrency["groq"]["in_flight"] == 0
    assert primary.active == 0
    assert health_state == {}
    assert metrics["providers"]["groq"]["attempts"] == 0
    assert metrics["providers"]["groq"]["streaming_attempts"] == 1
    assert metrics["providers"]["groq"]["streaming_cancellations"] == 1
    assert metrics["providers"]["groq"]["streaming_failures_by_category"] == {
        "cancelled": 1
    }
    assert metrics["streaming"]["cancellations"] == 1


def test_health_counts_pre_and_post_token_failures_and_success_resets() -> None:
    health = ProviderHealthManager(failure_threshold=3)
    providers = [
        MockStreamingProvider(
            "groq",
            error_stream(LLMTimeoutError("private")),
        ),
        MockStreamingProvider(
            "groq",
            chunk_stream(
                "groq",
                ["partial"],
                error=LLMUpstreamError("private"),
            ),
        ),
        MockStreamingProvider("groq", chunk_stream("groq", ["ok"])),
    ]

    async def exercise() -> tuple[dict, dict]:
        with pytest.raises(LLMTimeoutError):
            await collect(LLMRouter(providers[0], health_manager=health))
        with pytest.raises(LLMUpstreamError):
            await collect(LLMRouter(providers[1], health_manager=health))
        failed = await health.snapshot()
        await collect(LLMRouter(providers[2], health_manager=health))
        return failed, await health.snapshot()

    failed, reset = asyncio.run(exercise())

    assert failed["groq"]["consecutive_failures"] == 2
    assert reset["groq"]["consecutive_failures"] == 0


def test_streaming_telemetry_records_counts_and_no_content() -> None:
    telemetry = LLMTelemetry()
    primary = MockStreamingProvider(
        "groq",
        chunk_stream("groq", ["secret-one", "secret-two"]),
    )
    router = LLMRouter(primary, telemetry=telemetry)

    async def exercise() -> dict:
        await collect(router)
        return await telemetry.snapshot()

    snapshot = asyncio.run(exercise())
    serialized = json.dumps(snapshot)

    assert snapshot["providers"]["groq"]["streaming_attempts"] == 1
    assert snapshot["providers"]["groq"]["streaming_successes"] == 1
    assert snapshot["streaming"]["total"] == 1
    assert snapshot["streaming"]["successes"] == 1
    assert snapshot["streaming"]["average_time_to_first_token_seconds"] >= 0
    assert snapshot["streaming"]["average_duration_seconds"] >= 0
    assert "secret-one" not in serialized
    assert "secret-two" not in serialized


def test_openai_streaming_provider_uses_real_stream_flag() -> None:
    async def upstream_chunks():
        for content in ("Hel", "lo"):
            yield SimpleNamespace(
                model="openai/gpt-oss-20b",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=content),
                    )
                ],
            )

    create = AsyncMock(return_value=upstream_chunks())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=AsyncMock(),
    )
    provider = object.__new__(GroqProvider)
    provider.default_model = "openai/gpt-oss-20b"
    provider.client = client

    async def exercise() -> list[str]:
        return [chunk.content async for chunk in provider.stream_chat(MESSAGES)]

    chunks = asyncio.run(exercise())

    assert chunks == ["Hel", "lo"]
    create.assert_awaited_once_with(
        model="openai/gpt-oss-20b",
        messages=MESSAGES,
        stream=True,
    )


def test_provider_streaming_capability_audit() -> None:
    assert object.__new__(GroqProvider).capabilities.streaming is True
    assert object.__new__(GeminiProvider).capabilities.streaming is True
    assert object.__new__(CloudflareProvider).capabilities.streaming is True
    assert object.__new__(LLM7Provider).capabilities.streaming is True
    assert object.__new__(OllamaProvider).capabilities.streaming is False
    assert object.__new__(KiloProvider).capabilities.streaming is False

    openrouter = object.__new__(OpenRouterProvider)
    openrouter._capabilities = ProviderCapabilities(streaming=True)
    assert openrouter.capabilities.streaming is True
