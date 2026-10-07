from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
import time
from typing import Any

from app.llm.base import BaseLLMProvider
from app.llm.budget import RequestBudgetPolicy
from app.llm.concurrency import ProviderConcurrencyManager
from app.llm.exceptions import (
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.health import ProviderHealthManager
from app.llm.router import LLMRouter
from app.llm.task import TaskType
from app.llm.telemetry import LLMTelemetry


MESSAGES = [{"role": "user", "content": "benchmark"}]
ProviderAction = Callable[[], Awaitable[dict[str, str]]]


@dataclass(slots=True)
class ScenarioResult:
    name: str
    passed: bool
    latency_seconds: float
    selected_provider: str | None = None
    fallback_depth: int = 0
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class MockProvider(BaseLLMProvider):
    def __init__(
        self,
        name: str,
        actions: list[ProviderAction] | None = None,
        default_action: ProviderAction | None = None,
    ) -> None:
        self._name = name
        self._actions = deque(actions or [])
        self._default_action = default_action or self._success
        self.calls = 0
        self.active_calls = 0
        self.max_active_calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> dict[str, str]:
        self.calls += 1
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            action = self._actions.popleft() if self._actions else self._default_action
            result = await action()
            if model is not None:
                result = dict(result)
                result["model"] = model
            return result
        finally:
            self.active_calls -= 1

    async def close(self) -> None:
        return None

    async def _success(self) -> dict[str, str]:
        return {
            "provider": self.name,
            "model": f"mock/{self.name}",
            "content": "ok",
        }


def success_action(
    provider_name: str,
    delay: float = 0.0,
) -> ProviderAction:
    async def action() -> dict[str, str]:
        if delay:
            await asyncio.sleep(delay)
        return {
            "provider": provider_name,
            "model": f"mock/{provider_name}",
            "content": "ok",
        }

    return action


def error_action(error: Exception, delay: float = 0.0) -> ProviderAction:
    async def action() -> dict[str, str]:
        if delay:
            await asyncio.sleep(delay)
        raise error

    return action


def hanging_action() -> ProviderAction:
    async def action() -> dict[str, str]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    return action


def budget(seconds: float) -> RequestBudgetPolicy:
    return RequestBudgetPolicy(
        standard_seconds=seconds,
        reasoning_seconds=seconds,
        code_seconds=seconds,
        long_context_seconds=seconds,
        multimodal_seconds=seconds,
    )


async def run_mock_scenarios() -> dict[str, Any]:
    runners = (
        _healthy_primary,
        _recoverable_fallback,
        _circuit_breaker,
        _request_budget,
        _concurrency_bulkhead,
        _provider_independence,
        _semaphore_budget,
        _mixed_workload,
    )
    results: list[ScenarioResult] = []
    artifacts: dict[str, Any] = {}
    for runner in runners:
        result, artifact = await runner()
        results.append(result)
        artifacts[result.name] = artifact

    mixed = artifacts["mixed_resilience_workload"]
    summary = dict(mixed["summary"])
    summary["mixed_workload_max_observed_concurrency"] = summary[
        "max_observed_concurrency"
    ]
    summary["max_observed_concurrency"] = max(
        result.details.get("max_observed_concurrency", 0)
        if result.details
        else 0
        for result in results
    )
    summary["circuit_skips"] = artifacts["circuit_breaker"]["telemetry"][
        "providers"
    ]["groq"]["circuit_skips"]
    passed = all(result.passed for result in results)
    return {
        "mode": "mock",
        "passed": passed,
        "summary": summary,
        "scenarios": [result.to_dict() for result in results],
        "providers": mixed["telemetry"]["providers"],
        "telemetry": mixed["telemetry"],
        "telemetry_reconciliation": mixed["reconciliation"],
        "health": artifacts["circuit_breaker"]["health"],
        "concurrency": {
            "bulkhead": artifacts["concurrency_bulkhead"]["concurrency"],
            "independence": artifacts["provider_independence"]["concurrency"],
            "semaphore_budget": artifacts["semaphore_wait_budget"]["concurrency"],
        },
    }


async def _healthy_primary() -> tuple[ScenarioResult, dict[str, Any]]:
    telemetry = LLMTelemetry()
    health = ProviderHealthManager()
    primary = MockProvider("groq", default_action=success_action("groq", 0.05))
    router = LLMRouter(primary, health_manager=health, telemetry=telemetry)
    started = time.monotonic()
    response = await router.chat(MESSAGES)
    latency = time.monotonic() - started
    snapshot = await telemetry.snapshot()
    passed = (
        response["provider"] == "groq"
        and snapshot["requests"]["last_fallback_depth"] == 0
        and snapshot["providers"]["groq"]["successes"] == 1
    )
    return ScenarioResult(
        "healthy_primary",
        passed,
        latency,
        "groq",
        0,
    ), {"telemetry": snapshot, "health": await health.snapshot()}


async def _recoverable_fallback() -> tuple[ScenarioResult, dict[str, Any]]:
    telemetry = LLMTelemetry()
    primary = MockProvider(
        "groq",
        actions=[error_action(LLMTimeoutError("sanitized"))],
    )
    fallback = MockProvider("openrouter")
    router = LLMRouter(primary, fallback, telemetry=telemetry)
    started = time.monotonic()
    response = await router.chat(MESSAGES)
    latency = time.monotonic() - started
    snapshot = await telemetry.snapshot()
    passed = (
        response["provider"] == "openrouter"
        and snapshot["requests"]["last_fallback_depth"] == 1
        and snapshot["providers"]["groq"]["timeout_failures"] == 1
        and snapshot["providers"]["openrouter"]["successes"] == 1
    )
    return ScenarioResult(
        "recoverable_fallback",
        passed,
        latency,
        "openrouter",
        1,
    ), {"telemetry": snapshot}


async def _circuit_breaker() -> tuple[ScenarioResult, dict[str, Any]]:
    clock = FakeClock()
    telemetry = LLMTelemetry(clock=clock)
    health = ProviderHealthManager(
        failure_threshold=3,
        cooldown_seconds=60,
        clock=clock,
    )
    primary = MockProvider(
        "groq",
        actions=[
            error_action(LLMTimeoutError("one")),
            error_action(LLMTimeoutError("two")),
            error_action(LLMTimeoutError("three")),
            success_action("groq"),
        ],
    )
    fallback = MockProvider("openrouter")
    router = LLMRouter(
        primary,
        fallback,
        health_manager=health,
        telemetry=telemetry,
        clock=clock,
    )
    started = time.monotonic()
    for _ in range(3):
        await router.chat(MESSAGES)
    opened = await health.snapshot()
    calls_before_skip = primary.calls
    skipped_response = await router.chat(MESSAGES)
    calls_after_skip = primary.calls
    clock.advance(61)
    recovered_response = await router.chat(MESSAGES)
    latency = time.monotonic() - started
    telemetry_snapshot = await telemetry.snapshot()
    health_snapshot = await health.snapshot()
    passed = (
        opened["groq"]["circuit_open"] is True
        and calls_before_skip == calls_after_skip == 3
        and skipped_response["provider"] == "openrouter"
        and telemetry_snapshot["providers"]["groq"]["circuit_skips"] == 1
        and recovered_response["provider"] == "groq"
        and health_snapshot["groq"]["consecutive_failures"] == 0
        and health_snapshot["groq"]["circuit_open"] is False
    )
    return ScenarioResult(
        "circuit_breaker",
        passed,
        latency,
        "groq",
        0,
        {"circuit_opened": opened["groq"]["circuit_open"], "circuit_skips": 1},
    ), {"telemetry": telemetry_snapshot, "health": health_snapshot}


async def _request_budget() -> tuple[ScenarioResult, dict[str, Any]]:
    telemetry = LLMTelemetry()
    health = ProviderHealthManager(failure_threshold=3)
    primary = MockProvider("groq", default_action=hanging_action())
    fallback = MockProvider("openrouter")
    router = LLMRouter(
        primary,
        fallback,
        health_manager=health,
        budget_policy=budget(0.02),
        telemetry=telemetry,
    )
    started = time.monotonic()
    timed_out = False
    try:
        await router.chat(MESSAGES)
    except LLMTimeoutError:
        timed_out = True
    latency = time.monotonic() - started
    snapshot = await telemetry.snapshot()
    passed = (
        timed_out
        and fallback.calls == 0
        and snapshot["requests"]["failures"] == 1
        and snapshot["providers"]["groq"]["timeout_failures"] == 1
    )
    return ScenarioResult(
        "request_budget",
        passed,
        latency,
        None,
        1,
    ), {"telemetry": snapshot, "health": await health.snapshot()}


async def _concurrency_bulkhead() -> tuple[ScenarioResult, dict[str, Any]]:
    manager = ProviderConcurrencyManager(limits={"groq": 3})
    primary = MockProvider("groq", default_action=success_action("groq", 0.005))
    router = LLMRouter(primary, concurrency_manager=manager)
    started = time.monotonic()
    responses = await asyncio.gather(*(router.chat(MESSAGES) for _ in range(20)))
    latency = time.monotonic() - started
    snapshot = await manager.snapshot()
    state = snapshot["groq"]
    passed = (
        len(responses) == 20
        and primary.max_active_calls == 3
        and state["in_flight"] == 0
        and state["available"] == 3
    )
    return ScenarioResult(
        "concurrency_bulkhead",
        passed,
        latency,
        "groq",
        0,
        {"max_observed_concurrency": primary.max_active_calls},
    ), {"concurrency": snapshot}


async def _provider_independence() -> tuple[ScenarioResult, dict[str, Any]]:
    manager = ProviderConcurrencyManager(limits={"groq": 2, "openrouter": 3})
    groq = MockProvider("groq", default_action=success_action("groq", 0.005))
    openrouter = MockProvider(
        "openrouter",
        default_action=success_action("openrouter", 0.005),
    )
    groq_router = LLMRouter(groq, concurrency_manager=manager)
    openrouter_router = LLMRouter(openrouter, concurrency_manager=manager)
    started = time.monotonic()
    await asyncio.gather(
        *(groq_router.chat(MESSAGES) for _ in range(10)),
        *(openrouter_router.chat(MESSAGES) for _ in range(10)),
    )
    latency = time.monotonic() - started
    snapshot = await manager.snapshot()
    passed = (
        groq.max_active_calls == 2
        and openrouter.max_active_calls == 3
        and all(state["in_flight"] == 0 for state in snapshot.values())
    )
    return ScenarioResult(
        "provider_independence",
        passed,
        latency,
        None,
        0,
        {
            "groq_max": groq.max_active_calls,
            "openrouter_max": openrouter.max_active_calls,
        },
    ), {"concurrency": snapshot}


async def _semaphore_budget() -> tuple[ScenarioResult, dict[str, Any]]:
    manager = ProviderConcurrencyManager(limits={"groq": 1})
    health = ProviderHealthManager(failure_threshold=1)
    telemetry = LLMTelemetry()
    release = asyncio.Event()
    entered = asyncio.Event()

    async def blocked() -> dict[str, str]:
        entered.set()
        await release.wait()
        return {"provider": "groq", "model": "mock/groq", "content": "ok"}

    primary = MockProvider("groq", default_action=blocked)
    long_router = LLMRouter(
        primary,
        budget_policy=budget(1.0),
        concurrency_manager=manager,
    )
    short_router = LLMRouter(
        primary,
        health_manager=health,
        budget_policy=budget(0.02),
        telemetry=telemetry,
        concurrency_manager=manager,
    )
    started = time.monotonic()
    first = asyncio.create_task(long_router.chat(MESSAGES))
    await entered.wait()
    timed_out = False
    try:
        await short_router.chat(MESSAGES)
    except LLMTimeoutError:
        timed_out = True
    calls_after_timeout = primary.calls
    health_snapshot = await health.snapshot()
    telemetry_snapshot = await telemetry.snapshot()
    release.set()
    await first
    latency = time.monotonic() - started
    concurrency_snapshot = await manager.snapshot()
    passed = (
        timed_out
        and calls_after_timeout == 1
        and health_snapshot == {}
        and telemetry_snapshot["providers"] == {}
        and telemetry_snapshot["requests"]["failures"] == 1
        and concurrency_snapshot["groq"]["in_flight"] == 0
    )
    return ScenarioResult(
        "semaphore_wait_budget",
        passed,
        latency,
        None,
        0,
    ), {
        "telemetry": telemetry_snapshot,
        "health": health_snapshot,
        "concurrency": concurrency_snapshot,
    }


async def _mixed_workload() -> tuple[ScenarioResult, dict[str, Any]]:
    clock = FakeClock()
    telemetry = LLMTelemetry(clock=clock)
    health = ProviderHealthManager(failure_threshold=100, clock=clock)
    manager = ProviderConcurrencyManager(limits={"groq": 4, "openrouter": 2})

    def measured_success(provider_name: str) -> ProviderAction:
        async def action() -> dict[str, str]:
            clock.advance(0.02)
            return {
                "provider": provider_name,
                "model": f"mock/{provider_name}",
                "content": "ok",
            }

        return action

    def measured_error(error: Exception) -> ProviderAction:
        async def action() -> dict[str, str]:
            clock.advance(0.02)
            raise error

        return action

    primary_actions: list[ProviderAction] = []
    primary_actions.extend(measured_success("groq") for _ in range(30))
    primary_actions.extend(
        measured_error(LLMTimeoutError("timeout")) for _ in range(8)
    )
    primary_actions.extend(
        measured_error(LLMRateLimitError("rate")) for _ in range(5)
    )
    primary_actions.extend(
        measured_error(LLMUpstreamError("upstream")) for _ in range(3)
    )
    primary_actions.extend(
        measured_error(LLMProviderError("terminal")) for _ in range(2)
    )
    primary_actions.extend(
        measured_error(LLMTimeoutError("timeout")) for _ in range(2)
    )
    fallback_actions = [measured_success("openrouter") for _ in range(16)]
    fallback_actions.extend(
        measured_error(LLMUpstreamError("fallback upstream")) for _ in range(2)
    )
    primary = MockProvider("groq", actions=primary_actions)
    fallback = MockProvider("openrouter", actions=fallback_actions)
    router = LLMRouter(
        primary,
        fallback,
        health_manager=health,
        budget_policy=budget(1.0),
        telemetry=telemetry,
        concurrency_manager=manager,
        clock=clock,
    )
    successes = 0
    failures = 0
    max_depth = 0
    started = time.monotonic()
    for _ in range(50):
        try:
            await router.chat(MESSAGES)
            successes += 1
        except LLMProviderError:
            failures += 1
        current = await telemetry.snapshot()
        max_depth = max(max_depth, current["requests"]["last_fallback_depth"])
    latency = time.monotonic() - started
    telemetry_snapshot = await telemetry.snapshot()
    request_metrics = telemetry_snapshot["requests"]
    providers = telemetry_snapshot["providers"]
    expected = {
        "total_requests": 50,
        "successes": 46,
        "failures": 4,
        "groq_attempts": 50,
        "groq_successes": 30,
        "groq_timeouts": 10,
        "groq_rate_limits": 5,
        "groq_upstream_failures": 3,
        "groq_provider_failures": 2,
        "openrouter_attempts": 18,
        "openrouter_successes": 16,
        "openrouter_upstream_failures": 2,
        "total_fallback_depth": 22,
        "max_fallback_depth": 2,
    }
    actual = {
        "total_requests": request_metrics["total"],
        "successes": request_metrics["successes"],
        "failures": request_metrics["failures"],
        "groq_attempts": providers["groq"]["attempts"],
        "groq_successes": providers["groq"]["successes"],
        "groq_timeouts": providers["groq"]["timeout_failures"],
        "groq_rate_limits": providers["groq"]["rate_limit_failures"],
        "groq_upstream_failures": providers["groq"]["upstream_failures"],
        "groq_provider_failures": providers["groq"]["provider_failures"],
        "openrouter_attempts": providers["openrouter"]["attempts"],
        "openrouter_successes": providers["openrouter"]["successes"],
        "openrouter_upstream_failures": providers["openrouter"][
            "upstream_failures"
        ],
        "total_fallback_depth": request_metrics["total_fallback_depth"],
        "max_fallback_depth": max_depth,
    }
    reconciled = actual == expected and successes == 46 and failures == 4
    concurrency_snapshot = await manager.snapshot()
    slots_released = all(
        state["in_flight"] == 0 for state in concurrency_snapshot.values()
    )
    passed = reconciled and slots_released
    summary = {
        "total_requests": request_metrics["total"],
        "successes": request_metrics["successes"],
        "failures": request_metrics["failures"],
        "average_latency_seconds": request_metrics["average_latency_seconds"],
        "average_fallback_depth": request_metrics["average_fallback_depth"],
        "max_fallback_depth": max_depth,
        "circuit_skips": sum(
            item["circuit_skips"] for item in providers.values()
        ),
        "max_observed_concurrency": max(
            primary.max_active_calls,
            fallback.max_active_calls,
        ),
    }
    return ScenarioResult(
        "mixed_resilience_workload",
        passed,
        latency,
        None,
        max_depth,
        {"telemetry_reconciled": reconciled, "slots_released": slots_released},
    ), {
        "summary": summary,
        "telemetry": telemetry_snapshot,
        "health": await health.snapshot(),
        "concurrency": concurrency_snapshot,
        "reconciliation": {
            "passed": reconciled,
            "expected": expected,
            "actual": actual,
        },
    }
