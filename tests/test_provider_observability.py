import asyncio
import json

import pytest

from app.core.observability import JSONFormatter, ObservabilityMiddleware
from app.core.request_context import request_context
from app.llm.budget import RequestBudgetPolicy
from app.llm.capabilities import ProviderCapabilities
from app.llm.concurrency import ProviderConcurrencyManager
from app.llm.exceptions import LLMTimeoutError, LLMUpstreamError
from app.llm.health import ProviderHealthManager
from app.llm.router import LLMRouter
from app.llm.streaming import ProviderStreamChunk
from app.llm.telemetry import LLMTelemetry

SECRET = "TEST_SECRET_DO_NOT_LOG"
MESSAGES = [{"role": "user", "content": SECRET + "_PROMPT"}]


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


class Provider:
    capabilities = ProviderCapabilities(streaming=True)

    def __init__(self, name, clock, *, contents=("output",), error=None, entered=None):
        self.name = name
        self.clock = clock
        self.contents = contents
        self.error = error
        self.entered = entered
        self.calls = 0

    async def chat(self, messages, model=None):
        self.calls += 1
        self.clock.value += 0.25
        if self.error:
            raise self.error
        return {"provider": self.name, "model": "mock", "content": SECRET + "_RESPONSE"}

    async def stream_chat(self, messages, model=None):
        self.calls += 1
        for content in self.contents:
            self.clock.value += 0.25
            yield ProviderStreamChunk(self.name, "mock", content)
        if self.entered:
            self.entered.set()
            await asyncio.Event().wait()
        if self.error:
            raise self.error


def events(caplog):
    return [
        json.loads(JSONFormatter().format(record))
        for record in caplog.records
        if getattr(record, "event", None)
    ]


def test_provider_fallback_success_latency_and_correlation(caplog):
    clock = Clock()
    telemetry = LLMTelemetry(clock=clock)
    primary = Provider("groq", clock, error=LLMTimeoutError(SECRET + "_ERROR"))
    fallback = Provider("openrouter", clock)
    router = LLMRouter(primary, fallback, clock=clock, telemetry=telemetry)

    async def inner(scope, receive, send):
        assert request_context.get().request_id == "fallback-id"
        await router.chat(MESSAGES, model="mock")

    asyncio.run(
        ObservabilityMiddleware(inner)(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat",
                "headers": [(b"x-request-id", b"fallback-id")],
            },
            None,
            None,
        )
    )
    captured = events(caplog)
    provider_events = [
        event for event in captured if event["event"].startswith("provider_attempt")
    ]
    assert [event["event"] for event in provider_events] == [
        "provider_attempt_started",
        "provider_attempt_failed",
        "provider_attempt_started",
        "provider_attempt_completed",
    ]
    assert all(event["request_id"] == "fallback-id" for event in provider_events)
    assert provider_events[1]["error_category"] == "timeout"
    assert provider_events[1]["duration_ms"] == 250
    assert provider_events[-1]["fallback_depth"] == 1
    assert SECRET not in json.dumps(captured) + caplog.text
    snapshot = asyncio.run(telemetry.snapshot())
    assert snapshot["providers"]["groq"]["timeout_failures"] == 1
    assert snapshot["providers"]["openrouter"]["successes"] == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_circuit_skip_does_not_start_attempt(streaming, caplog):
    clock = Clock()
    health = ProviderHealthManager(failure_threshold=1, clock=clock)
    telemetry = LLMTelemetry()
    primary = Provider("groq", clock)
    fallback = Provider("openrouter", clock)
    router = LLMRouter(
        primary, fallback, health_manager=health, telemetry=telemetry, clock=clock
    )

    async def exercise():
        await health.record_failure("groq", LLMTimeoutError(SECRET))
        if streaming:
            return [event async for event in router.stream_chat(MESSAGES)]
        return await router.chat(MESSAGES)

    asyncio.run(exercise())
    assert primary.calls == 0
    assert fallback.calls == 1
    captured = events(caplog)
    assert any(
        event["event"] == "provider_circuit_skipped" and event["provider"] == "groq"
        for event in captured
    )
    assert not any(
        event["event"] == "provider_attempt_started" and event["provider"] == "groq"
        for event in captured
    )
    assert asyncio.run(telemetry.snapshot())["providers"]["groq"]["circuit_skips"] == 1


def test_selected_attempt_ttft_ignores_empty_chunks_and_failed_provider(caplog):
    clock = Clock()
    primary = Provider("groq", clock, contents=("", ""), error=LLMUpstreamError(SECRET))
    fallback = Provider("openrouter", clock, contents=("", SECRET + "_DELTA", "two"))
    telemetry = LLMTelemetry()
    router = LLMRouter(primary, fallback, clock=clock, telemetry=telemetry)

    async def exercise():
        emitted = [event async for event in router.stream_chat(MESSAGES)]
        return emitted, await telemetry.snapshot()

    emitted, snapshot = asyncio.run(exercise())
    assert [event.event for event in emitted] == ["meta", "delta", "delta", "done"]
    assert emitted[0].data["provider"] == "openrouter"
    assert snapshot["streaming"]["last_time_to_first_token_seconds"] == 0.5
    assert snapshot["streaming"]["last_duration_seconds"] == 1.25
    assert snapshot["streaming"]["last_selected_provider"] == "openrouter"
    assert snapshot["providers"]["groq"]["streaming_failures_by_category"] == {
        "upstream": 1
    }
    assert snapshot["providers"]["openrouter"]["streaming_last_latency_seconds"] == 0.75
    captured = events(caplog)
    first = [event for event in captured if event["event"] == "stream_first_token"]
    assert len(first) == 1
    assert first[0]["ttft_ms"] == 500
    assert first[0]["provider"] == "openrouter"
    assert sum(event["event"] == "stream_completed" for event in captured) == 1
    assert SECRET not in json.dumps(captured)


def test_post_first_delta_failure_has_no_fallback_and_no_per_token_logs(caplog):
    clock = Clock()
    primary = Provider(
        "groq", clock, contents=(SECRET,) * 20, error=LLMUpstreamError(SECRET)
    )
    fallback = Provider("openrouter", clock)
    telemetry = LLMTelemetry()
    router = LLMRouter(primary, fallback, telemetry=telemetry, clock=clock)

    async def exercise():
        emitted = []
        with pytest.raises(LLMUpstreamError):
            async for event in router.stream_chat(MESSAGES):
                emitted.append(event)
        return emitted

    emitted = asyncio.run(exercise())
    assert len(emitted) == 21
    assert fallback.calls == 0
    assert emitted[-1].event == "delta"
    captured = events(caplog)
    assert len(captured) == 5
    assert captured[-1]["event"] == "stream_failed"
    assert captured[-1]["error_category"] == "upstream"
    assert SECRET not in json.dumps(captured)


@pytest.mark.parametrize("after_token", [False, True])
def test_stream_cancellation_logs_and_metrics_are_safe(after_token, caplog):
    async def exercise():
        clock = Clock()
        entered = asyncio.Event()
        primary = Provider(
            "groq", clock, contents=(SECRET,) if after_token else (), entered=entered
        )
        fallback = Provider("openrouter", clock)
        telemetry = LLMTelemetry()
        router = LLMRouter(primary, fallback, clock=clock, telemetry=telemetry)

        async def consume():
            async for _ in router.stream_chat(MESSAGES):
                pass

        task = asyncio.create_task(consume())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fallback.calls == 0
        return await telemetry.snapshot()

    snapshot = asyncio.run(exercise())
    assert snapshot["streaming"]["cancellations"] == 1
    assert snapshot["providers"]["groq"]["streaming_cancellations"] == 1
    assert any(event["event"] == "stream_cancelled" for event in events(caplog))
    assert SECRET not in json.dumps(events(caplog))


def test_stream_snapshot_clears_stale_ttft_and_is_detached():
    async def exercise():
        clock = Clock()
        telemetry = LLMTelemetry()
        router = LLMRouter(Provider("groq", clock), clock=clock, telemetry=telemetry)
        _ = [event async for event in router.stream_chat(MESSAGES)]
        router.primary_provider = Provider(
            "groq", clock, contents=(), error=LLMUpstreamError(SECRET)
        )
        with pytest.raises(LLMUpstreamError):
            _ = [event async for event in router.stream_chat(MESSAGES)]
        snapshot = await telemetry.snapshot()
        assert snapshot["streaming"]["last_time_to_first_token_seconds"] is None
        snapshot["providers"]["groq"]["streaming_failures_by_category"]["upstream"] = (
            999
        )
        return await telemetry.snapshot()

    snapshot = asyncio.run(exercise())
    assert snapshot["providers"]["groq"]["streaming_failures_by_category"] == {
        "upstream": 1
    }


def test_budget_expiration_event_is_distinct_from_provider_timeout(caplog):
    async def exercise():
        clock = Clock()
        primary = Provider("groq", clock, contents=(), entered=asyncio.Event())
        router = LLMRouter(
            primary,
            clock=clock,
            budget_policy=RequestBudgetPolicy(standard_seconds=0.001),
            telemetry=LLMTelemetry(),
        )
        with pytest.raises(LLMTimeoutError):
            _ = [event async for event in router.stream_chat(MESSAGES)]

    asyncio.run(exercise())
    assert any(
        event.get("error_category") == "budget_timeout" for event in events(caplog)
    )


def test_concurrent_provider_logs_keep_request_id(caplog):
    async def exercise():
        both_entered = asyncio.Event()
        release = asyncio.Event()
        count = 0

        class BlockingProvider:
            name = "groq"

            async def chat(self, messages):
                nonlocal count
                count += 1
                if count == 2:
                    both_entered.set()
                await release.wait()
                return {"provider": "groq", "model": "mock", "content": SECRET}

        router = LLMRouter(BlockingProvider())

        async def inner(scope, receive, send):
            await router.chat(MESSAGES)

        middleware = ObservabilityMiddleware(inner)
        tasks = [
            asyncio.create_task(
                middleware(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/v1/chat",
                        "headers": [(b"x-request-id", request_id)],
                    },
                    None,
                    None,
                )
            )
            for request_id in (b"first", b"second")
        ]
        await both_entered.wait()
        release.set()
        await asyncio.gather(*tasks)
        assert request_context.get() is None

    asyncio.run(exercise())
    captured = [
        event
        for event in events(caplog)
        if event["event"].startswith("provider_attempt")
    ]
    assert len(captured) == 4
    assert {event["request_id"] for event in captured} == {"first", "second"}
    for request_id in ("first", "second"):
        assert [
            event["event"] for event in captured if event["request_id"] == request_id
        ] == ["provider_attempt_started", "provider_attempt_completed"]


def test_streaming_failure_categories_are_bounded():
    async def exercise():
        telemetry = LLMTelemetry()
        await telemetry.record_provider_stream_result(
            "groq", success=False, error_category=SECRET
        )
        return await telemetry.snapshot()

    snapshot = asyncio.run(exercise())
    assert snapshot["providers"]["groq"]["streaming_failures_by_category"] == {
        "provider": 1
    }
    assert SECRET not in json.dumps(snapshot)


def test_cancellation_while_waiting_for_stream_slot_never_attempts_or_falls_back(
    caplog,
):
    async def exercise():
        clock = Clock()
        telemetry = LLMTelemetry()
        concurrency = ProviderConcurrencyManager(limits={"groq": 1})
        primary = Provider("groq", clock)
        fallback = Provider("openrouter", clock)
        waiting = asyncio.Event()

        class ObservedConcurrency:
            def slot(self, provider, timeout_seconds):
                waiting.set()
                return concurrency.slot(provider, timeout_seconds=timeout_seconds)

        router = LLMRouter(
            primary,
            fallback,
            clock=clock,
            telemetry=telemetry,
            concurrency_manager=ObservedConcurrency(),
        )

        async def consume():
            async for _ in router.stream_chat(MESSAGES):
                pass

        async with concurrency.slot("groq"):
            task = asyncio.create_task(consume())
            await waiting.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert primary.calls == fallback.calls == 0
        assert (await concurrency.snapshot())["groq"]["in_flight"] == 0
        return await telemetry.snapshot()

    snapshot = asyncio.run(exercise())
    assert snapshot["providers"] == {}
    assert snapshot["streaming"]["cancellations"] == 1
    assert [event["event"] for event in events(caplog)] == [
        "stream_started",
        "stream_cancelled",
    ]
