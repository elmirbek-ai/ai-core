import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.llm.budget import RequestBudgetPolicy
from app.llm.concurrency import (
    ProviderConcurrencyManager,
    ProviderConcurrencyTimeout,
)
from app.llm.exceptions import LLMTimeoutError
from app.llm.health import ProviderHealthManager
from app.llm.router import LLMRouter
from app.llm.telemetry import LLMTelemetry


MESSAGES = [{"role": "user", "content": "Hello"}]
GROQ_RESULT = {
    "provider": "groq",
    "model": "openai/gpt-oss-20b",
    "content": "primary response",
}
OPENROUTER_RESULT = {
    "provider": "openrouter",
    "model": "openrouter/free",
    "content": "fallback response",
}


def provider(name: str, *, result=None, error=None):
    return SimpleNamespace(
        name=name,
        chat=AsyncMock(return_value=result, side_effect=error),
        close=AsyncMock(),
    )


def budget(seconds: float) -> RequestBudgetPolicy:
    return RequestBudgetPolicy(
        standard_seconds=seconds,
        reasoning_seconds=seconds,
        code_seconds=seconds,
        long_context_seconds=seconds,
        multimodal_seconds=seconds,
    )


def test_first_request_acquires_slot_and_success_releases_it() -> None:
    manager = ProviderConcurrencyManager(limits={"groq": 1})

    async def exercise() -> tuple[dict, dict]:
        async with manager.slot("groq"):
            during = await manager.snapshot()
        return during, await manager.snapshot()

    during, after = asyncio.run(exercise())

    assert during["groq"] == {"limit": 1, "in_flight": 1, "available": 0}
    assert after["groq"] == {"limit": 1, "in_flight": 0, "available": 1}


def test_slot_released_after_error() -> None:
    manager = ProviderConcurrencyManager(limits={"groq": 1})

    async def exercise() -> dict:
        with pytest.raises(RuntimeError):
            async with manager.slot("groq"):
                raise RuntimeError("provider failed")
        return await manager.snapshot()

    snapshot = asyncio.run(exercise())

    assert snapshot["groq"]["in_flight"] == 0
    assert snapshot["groq"]["available"] == 1


def test_slot_released_after_cancellation() -> None:
    manager = ProviderConcurrencyManager(limits={"groq": 1})

    async def exercise() -> dict:
        entered = asyncio.Event()

        async def hold_slot() -> None:
            async with manager.slot("groq"):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(hold_slot())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return await manager.snapshot()

    snapshot = asyncio.run(exercise())

    assert snapshot["groq"]["in_flight"] == 0
    assert snapshot["groq"]["available"] == 1


def test_wait_timeout_does_not_leak_slot() -> None:
    manager = ProviderConcurrencyManager(limits={"groq": 1})

    async def exercise() -> tuple[dict, dict]:
        async with manager.slot("groq"):
            with pytest.raises(ProviderConcurrencyTimeout):
                async with manager.slot("groq", timeout_seconds=0.01):
                    pytest.fail("waiting request must not acquire")
            during = await manager.snapshot()
        return during, await manager.snapshot()

    during, after = asyncio.run(exercise())

    assert during["groq"]["in_flight"] == 1
    assert after["groq"]["in_flight"] == 0
    assert after["groq"]["available"] == 1


def test_fifth_request_waits_at_limit_four_then_proceeds() -> None:
    manager = ProviderConcurrencyManager(limits={"groq": 4})

    async def exercise() -> tuple[bool, dict, dict]:
        release = asyncio.Event()
        four_entered = asyncio.Event()
        entered_count = 0

        async def holder() -> None:
            nonlocal entered_count
            async with manager.slot("groq"):
                entered_count += 1
                if entered_count == 4:
                    four_entered.set()
                await release.wait()

        holders = [asyncio.create_task(holder()) for _ in range(4)]
        await four_entered.wait()
        fifth_entered = asyncio.Event()

        async def fifth() -> None:
            async with manager.slot("groq"):
                fifth_entered.set()

        fifth_task = asyncio.create_task(fifth())
        await asyncio.sleep(0)
        waited = not fifth_entered.is_set()
        during = await manager.snapshot()
        release.set()
        await asyncio.gather(*holders, fifth_task)
        return waited, during, await manager.snapshot()

    waited, during, after = asyncio.run(exercise())

    assert waited is True
    assert during["groq"]["in_flight"] == 4
    assert during["groq"]["available"] == 0
    assert after["groq"]["in_flight"] == 0


def test_provider_limits_are_independent() -> None:
    manager = ProviderConcurrencyManager(
        limits={"groq": 1, "openrouter": 1},
    )

    async def exercise() -> dict:
        async with manager.slot("groq"):
            async with manager.slot("openrouter", timeout_seconds=0.01):
                return await manager.snapshot()

    snapshot = asyncio.run(exercise())

    assert snapshot["groq"]["in_flight"] == 1
    assert snapshot["openrouter"]["in_flight"] == 1


def test_feature_disabled_bypasses_limit() -> None:
    manager = ProviderConcurrencyManager(
        enabled=False,
        limits={"groq": 1},
    )

    async def exercise() -> tuple[int, dict]:
        entered = 0
        both_entered = asyncio.Event()
        release = asyncio.Event()

        async def worker() -> None:
            nonlocal entered
            async with manager.slot("groq"):
                entered += 1
                if entered == 2:
                    both_entered.set()
                await release.wait()

        tasks = [asyncio.create_task(worker()) for _ in range(2)]
        await both_entered.wait()
        observed = entered
        release.set()
        await asyncio.gather(*tasks)
        return observed, await manager.snapshot()

    observed, snapshot = asyncio.run(exercise())

    assert observed == 2
    assert snapshot["groq"]["in_flight"] == 0


@pytest.mark.parametrize(
    "field",
    [
        "llm_provider_default_max_concurrency",
        "llm_groq_max_concurrency",
        "llm_openrouter_max_concurrency",
        "llm_gemini_max_concurrency",
        "llm_cloudflare_max_concurrency",
        "llm_ollama_max_concurrency",
        "llm_kilo_max_concurrency",
        "llm_llm7_max_concurrency",
    ],
)
def test_invalid_config_limit_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            groq_api_key="test-key",
            **{field: 0},
        )


def test_config_defaults() -> None:
    settings = Settings(_env_file=None, groq_api_key="test-key")

    assert settings.llm_provider_concurrency_enabled is True
    assert settings.llm_provider_default_max_concurrency == 4
    assert settings.llm_groq_max_concurrency == 4
    assert settings.llm_openrouter_max_concurrency == 2
    assert settings.llm_gemini_max_concurrency == 3
    assert settings.llm_cloudflare_max_concurrency == 3
    assert settings.llm_ollama_max_concurrency == 2
    assert settings.llm_kilo_max_concurrency == 1
    assert settings.llm_llm7_max_concurrency == 1


def test_snapshot_does_not_leak_mutable_internal_state() -> None:
    manager = ProviderConcurrencyManager(limits={"groq": 2})

    async def exercise() -> tuple[dict, dict]:
        first = await manager.snapshot()
        first["groq"]["in_flight"] = 99
        return first, await manager.snapshot()

    _, second = asyncio.run(exercise())

    assert second["groq"] == {"limit": 2, "in_flight": 0, "available": 2}


def test_circuit_open_provider_does_not_acquire_or_call() -> None:
    health = ProviderHealthManager(failure_threshold=1)
    manager = ProviderConcurrencyManager(
        limits={"groq": 1, "openrouter": 1},
    )
    primary = provider("groq", result=GROQ_RESULT)
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary,
        fallback,
        health_manager=health,
        concurrency_manager=manager,
    )

    async def exercise() -> tuple[dict[str, str], dict]:
        await health.record_failure("groq", LLMTimeoutError("private"))
        result = await router.chat(MESSAGES)
        return result, await manager.snapshot()

    result, snapshot = asyncio.run(exercise())

    assert result == OPENROUTER_RESULT
    assert snapshot["groq"]["in_flight"] == 0
    primary.chat.assert_not_awaited()


def test_disabled_provider_never_acquires_slot() -> None:
    manager = ProviderConcurrencyManager(
        limits={"groq": 1, "openrouter": 1},
    )
    primary = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(
        primary_provider=primary,
        fallback_provider=None,
        concurrency_manager=manager,
    )

    result = asyncio.run(router.chat(MESSAGES))
    snapshot = asyncio.run(manager.snapshot())

    assert result == GROQ_RESULT
    assert snapshot["openrouter"]["in_flight"] == 0


def test_slot_wait_deadline_is_request_only_failure() -> None:
    manager = ProviderConcurrencyManager(limits={"groq": 1})
    health = ProviderHealthManager(failure_threshold=1)
    telemetry = LLMTelemetry()
    release = asyncio.Event()
    entered = asyncio.Event()

    async def block_first(messages, model=None):
        entered.set()
        await release.wait()
        return GROQ_RESULT

    primary = provider("groq")
    primary.chat.side_effect = block_first
    long_router = LLMRouter(
        primary,
        budget_policy=budget(1.0),
        concurrency_manager=manager,
    )
    short_router = LLMRouter(
        primary,
        health_manager=health,
        budget_policy=budget(0.01),
        telemetry=telemetry,
        concurrency_manager=manager,
    )

    async def exercise() -> tuple[dict, dict, dict[str, str]]:
        first = asyncio.create_task(long_router.chat(MESSAGES))
        await entered.wait()
        with pytest.raises(LLMTimeoutError):
            await short_router.chat(MESSAGES)
        call_count_after_timeout = primary.chat.await_count
        health_snapshot = await health.snapshot()
        telemetry_snapshot = await telemetry.snapshot()
        release.set()
        first_result = await first
        return (
            health_snapshot,
            telemetry_snapshot,
            {
                "calls": str(call_count_after_timeout),
                "provider": first_result["provider"],
            },
        )

    health_snapshot, telemetry_snapshot, outcome = asyncio.run(exercise())

    assert outcome == {"calls": "1", "provider": "groq"}
    assert health_snapshot == {}
    assert telemetry_snapshot["providers"] == {}
    assert telemetry_snapshot["requests"]["failures"] == 1
    assert telemetry_snapshot["requests"]["last_error_category"] == "timeout"


def test_provider_timeout_releases_slot_and_counts_attempt() -> None:
    async def hang(messages, model=None):
        await asyncio.Event().wait()

    manager = ProviderConcurrencyManager(limits={"groq": 1})
    health = ProviderHealthManager(failure_threshold=3)
    telemetry = LLMTelemetry()
    primary = provider("groq")
    primary.chat.side_effect = hang
    router = LLMRouter(
        primary,
        health_manager=health,
        budget_policy=budget(0.01),
        telemetry=telemetry,
        concurrency_manager=manager,
    )

    async def exercise() -> tuple[dict, dict, dict]:
        with pytest.raises(LLMTimeoutError):
            await router.chat(MESSAGES)
        return (
            await manager.snapshot(),
            await health.snapshot(),
            await telemetry.snapshot(),
        )

    concurrency, health_state, metrics = asyncio.run(exercise())

    assert concurrency["groq"]["available"] == 1
    assert health_state["groq"]["consecutive_failures"] == 1
    assert metrics["providers"]["groq"]["attempts"] == 1
    assert metrics["providers"]["groq"]["timeout_failures"] == 1


def test_one_hundred_requests_never_exceed_limit_five() -> None:
    manager = ProviderConcurrencyManager(limits={"groq": 5})
    active = 0
    max_observed = 0

    async def tracked_chat(messages, model=None):
        nonlocal active, max_observed
        active += 1
        max_observed = max(max_observed, active)
        await asyncio.sleep(0.001)
        active -= 1
        return GROQ_RESULT

    primary = provider("groq")
    primary.chat.side_effect = tracked_chat
    router = LLMRouter(primary, concurrency_manager=manager)

    async def exercise() -> tuple[list[dict[str, str]], dict]:
        results = await asyncio.gather(
            *(router.chat(MESSAGES) for _ in range(100)),
        )
        return results, await manager.snapshot()

    results, snapshot = asyncio.run(exercise())

    assert len(results) == 100
    assert max_observed == 5
    assert snapshot["groq"]["in_flight"] == 0
    assert snapshot["groq"]["available"] == 5
