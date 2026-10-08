import asyncio

import pytest

from app.llm.base import BaseLLMProvider
from app.llm.capabilities import ProviderCapabilities
from app.llm.concurrency import ProviderConcurrencyManager
from app.llm.health import ProviderHealthManager
from app.llm.router import LLMRouter
from app.llm.streaming import ProviderStreamChunk, StreamEvent
from app.llm.telemetry import LLMTelemetry

MESSAGES = [{"role": "user", "content": "cancel safely"}]


class BlockingProvider(BaseLLMProvider):
    def __init__(self, name: str = "groq", *, emit_first_chunk: bool = False) -> None:
        self._name = name
        self.emit_first_chunk = emit_first_chunk
        self.entered = asyncio.Event()
        self.after_first_chunk = asyncio.Event()
        self.release = asyncio.Event()
        self.closed_call = asyncio.Event()
        self.chat_calls = 0
        self.stream_calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True)

    async def chat(self, messages, model=None):
        del messages, model
        self.chat_calls += 1
        self.entered.set()
        try:
            await self.release.wait()
        finally:
            self.closed_call.set()
        return {"provider": self.name, "model": "model", "content": "late"}

    async def stream_chat(self, messages, model=None):
        del messages, model
        self.stream_calls += 1
        self.entered.set()
        try:
            if self.emit_first_chunk:
                yield ProviderStreamChunk(self.name, "model", "partial")
                self.after_first_chunk.set()
            await self.release.wait()
            yield ProviderStreamChunk(self.name, "model", "late")
        finally:
            self.closed_call.set()

    async def close(self) -> None:
        return None


async def _collect_stream(router: LLMRouter, events: list[StreamEvent]) -> None:
    async for event in router.stream_chat(MESSAGES):
        events.append(event)


def test_non_stream_provider_cancellation_releases_slot_without_failure_state() -> None:
    async def exercise() -> tuple[dict, dict, dict, bool]:
        provider = BlockingProvider()
        concurrency = ProviderConcurrencyManager(limits={"groq": 1})
        health = ProviderHealthManager(failure_threshold=1)
        telemetry = LLMTelemetry()
        router = LLMRouter(
            provider,
            concurrency_manager=concurrency,
            health_manager=health,
            telemetry=telemetry,
        )

        task = asyncio.create_task(router.chat(MESSAGES))
        await provider.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return (
            await concurrency.snapshot(),
            await health.snapshot(),
            await telemetry.snapshot(),
            provider.closed_call.is_set(),
        )

    concurrency, health, telemetry, provider_closed = asyncio.run(exercise())

    assert concurrency["groq"] == {"limit": 1, "in_flight": 0, "available": 1}
    assert health == {}
    assert telemetry["providers"] == {}
    assert telemetry["requests"]["total"] == 0
    assert provider_closed


def test_stream_cancellation_before_first_token_closes_without_fallback() -> None:
    async def exercise() -> tuple[list[StreamEvent], dict, dict, dict, int, bool]:
        primary = BlockingProvider()
        fallback = BlockingProvider("openrouter", emit_first_chunk=True)
        concurrency = ProviderConcurrencyManager(limits={"groq": 1})
        health = ProviderHealthManager(failure_threshold=1)
        telemetry = LLMTelemetry()
        router = LLMRouter(
            primary,
            fallback,
            concurrency_manager=concurrency,
            health_manager=health,
            telemetry=telemetry,
        )
        events: list[StreamEvent] = []
        task = asyncio.create_task(_collect_stream(router, events))
        await primary.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return (
            events,
            await concurrency.snapshot(),
            await health.snapshot(),
            await telemetry.snapshot(),
            fallback.stream_calls,
            primary.closed_call.is_set(),
        )

    events, concurrency, health, telemetry, fallback_calls, closed = asyncio.run(
        exercise()
    )

    assert events == []
    assert fallback_calls == 0
    assert closed
    assert concurrency["groq"]["in_flight"] == 0
    assert health == {}
    assert telemetry["providers"] == {}
    assert telemetry["streaming"]["cancellations"] == 1


def test_stream_cancellation_after_delta_never_falls_back_or_emits_done() -> None:
    async def exercise() -> tuple[list[StreamEvent], dict, dict, dict, int, bool]:
        primary = BlockingProvider(emit_first_chunk=True)
        fallback = BlockingProvider("openrouter", emit_first_chunk=True)
        concurrency = ProviderConcurrencyManager(limits={"groq": 1})
        health = ProviderHealthManager(failure_threshold=1)
        telemetry = LLMTelemetry()
        router = LLMRouter(
            primary,
            fallback,
            concurrency_manager=concurrency,
            health_manager=health,
            telemetry=telemetry,
        )
        events: list[StreamEvent] = []
        task = asyncio.create_task(_collect_stream(router, events))
        await primary.after_first_chunk.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return (
            events,
            await concurrency.snapshot(),
            await health.snapshot(),
            await telemetry.snapshot(),
            fallback.stream_calls,
            primary.closed_call.is_set(),
        )

    events, concurrency, health, telemetry, fallback_calls, closed = asyncio.run(
        exercise()
    )

    assert [event.event for event in events] == ["meta", "delta"]
    assert "done" not in {event.event for event in events}
    assert fallback_calls == 0
    assert closed
    assert concurrency["groq"]["in_flight"] == 0
    assert health == {}
    assert telemetry["providers"] == {}
    assert telemetry["streaming"]["cancellations"] == 1
