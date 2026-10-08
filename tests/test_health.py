import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.health import ProviderHealthManager
from app.llm.router import LLMRouter

MESSAGES = [{"role": "user", "content": "private prompt"}]
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


class FakeClock:
    def __init__(self, current: float = 100.0) -> None:
        self.current = current

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


def provider(name: str, *, result=None, error=None):
    return SimpleNamespace(
        name=name,
        chat=AsyncMock(return_value=result, side_effect=error),
    )


def state_for(snapshot: dict, provider_name: str = "groq") -> dict:
    return snapshot[provider_name]


def test_initial_provider_is_available() -> None:
    manager = ProviderHealthManager()

    async def exercise() -> None:
        assert await manager.is_available("groq") is True
        assert await manager.snapshot() == {}

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("failure_count", "expected_open"),
    [(1, False), (2, False), (3, True)],
)
def test_recoverable_failures_open_at_threshold(
    failure_count: int,
    expected_open: bool,
) -> None:
    manager = ProviderHealthManager(failure_threshold=3)

    async def exercise() -> dict:
        for _ in range(failure_count):
            await manager.record_failure("groq", LLMTimeoutError("private"))
        return await manager.snapshot()

    state = state_for(asyncio.run(exercise()))

    assert state["consecutive_failures"] == failure_count
    assert state["circuit_open"] is expected_open
    assert state["last_failure_type"] == "LLMTimeoutError"


def test_open_circuit_is_unavailable_until_cooldown_expires() -> None:
    clock = FakeClock()
    manager = ProviderHealthManager(
        failure_threshold=1,
        cooldown_seconds=60,
        clock=clock,
    )

    async def exercise() -> tuple[bool, bool]:
        await manager.record_failure("groq", LLMUpstreamError("private"))
        before = await manager.is_available("groq")
        clock.advance(60)
        after = await manager.is_available("groq")
        return before, after

    assert asyncio.run(exercise()) == (False, True)


def test_success_resets_failures() -> None:
    manager = ProviderHealthManager(failure_threshold=3)

    async def exercise() -> dict:
        await manager.record_failure("groq", LLMTimeoutError("private"))
        await manager.record_success("groq")
        return await manager.snapshot()

    state = state_for(asyncio.run(exercise()))

    assert state["consecutive_failures"] == 0
    assert state["circuit_open"] is False
    assert state["last_failure_type"] is None
    assert state["last_failure_time"] is None


def test_success_after_cooldown_resets_open_state() -> None:
    clock = FakeClock()
    manager = ProviderHealthManager(
        failure_threshold=1,
        cooldown_seconds=10,
        clock=clock,
    )

    async def exercise() -> dict:
        await manager.record_failure("groq", LLMTimeoutError("private"))
        clock.advance(10)
        assert await manager.is_available("groq") is True
        await manager.record_success("groq")
        return await manager.snapshot()

    state = state_for(asyncio.run(exercise()))

    assert state["consecutive_failures"] == 0
    assert state["circuit_open"] is False


def test_provider_states_are_independent() -> None:
    manager = ProviderHealthManager(failure_threshold=1)

    async def exercise() -> tuple[dict, bool]:
        await manager.record_failure("groq", LLMTimeoutError("private"))
        return await manager.snapshot(), await manager.is_available("openrouter")

    snapshot, openrouter_available = asyncio.run(exercise())

    assert state_for(snapshot)["circuit_open"] is True
    assert "openrouter" not in snapshot
    assert openrouter_available is True


@pytest.mark.parametrize(
    "error",
    [
        LLMAuthenticationError("private auth"),
        LLMProviderError("private provider error"),
        ValueError("private config error"),
    ],
)
def test_non_recoverable_errors_are_not_counted(error: Exception) -> None:
    manager = ProviderHealthManager(failure_threshold=1)

    async def exercise() -> tuple[dict, bool]:
        await manager.record_failure("groq", error)
        return await manager.snapshot(), await manager.is_available("groq")

    snapshot, available = asyncio.run(exercise())

    assert snapshot == {}
    assert available is True


def test_concurrent_failure_updates_are_not_lost() -> None:
    manager = ProviderHealthManager(failure_threshold=200)

    async def exercise() -> dict:
        await asyncio.gather(
            *(
                manager.record_failure(
                    "groq",
                    LLMTimeoutError("private"),
                )
                for _ in range(100)
            )
        )
        return await manager.snapshot()

    state = state_for(asyncio.run(exercise()))

    assert state["consecutive_failures"] == 100
    assert state["circuit_open"] is False


def test_health_snapshot_mutation_cannot_corrupt_internal_state() -> None:
    manager = ProviderHealthManager(failure_threshold=3)

    async def exercise() -> tuple[dict, dict]:
        await manager.record_failure("groq", LLMTimeoutError("private"))
        first = await manager.snapshot()
        first["groq"]["consecutive_failures"] = 999
        first["injected"] = {"consecutive_failures": 999}
        return first, await manager.snapshot()

    _, second = asyncio.run(exercise())

    assert second["groq"]["consecutive_failures"] == 1
    assert "injected" not in second


def test_disabled_manager_never_tracks_or_skips() -> None:
    manager = ProviderHealthManager(enabled=False, failure_threshold=1)

    async def exercise() -> tuple[dict, bool]:
        await manager.record_failure("groq", LLMTimeoutError("private"))
        return await manager.snapshot(), await manager.is_available("groq")

    snapshot, available = asyncio.run(exercise())

    assert snapshot == {}
    assert available is True


def test_health_logs_do_not_include_error_or_prompt(caplog) -> None:
    manager = ProviderHealthManager(failure_threshold=1)
    secret = "private-sdk-error"

    with caplog.at_level(logging.INFO, logger="app.llm.health"):
        asyncio.run(
            manager.record_failure("groq", LLMTimeoutError(secret)),
        )

    assert "Provider groq circuit opened" in caplog.text
    assert secret not in caplog.text
    assert "private prompt" not in caplog.text


def test_config_defaults_and_validation() -> None:
    settings = Settings(_env_file=None, groq_api_key="test-key")

    assert settings.llm_circuit_breaker_enabled is True
    assert settings.llm_circuit_failure_threshold == 3
    assert settings.llm_circuit_cooldown_seconds == 60.0

    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            groq_api_key="test-key",
            llm_circuit_failure_threshold=0,
        )
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            groq_api_key="test-key",
            llm_circuit_cooldown_seconds=-1,
        )


@pytest.mark.parametrize(
    "error",
    [
        LLMTimeoutError("private"),
        LLMRateLimitError("private"),
        LLMUpstreamError("private"),
    ],
)
def test_router_records_each_recoverable_error(error: Exception) -> None:
    manager = ProviderHealthManager(failure_threshold=3)
    primary = provider("groq", error=error)
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary,
        fallback,
        health_manager=manager,
    )

    async def exercise() -> dict:
        await router.chat(MESSAGES)
        return await manager.snapshot()

    state = state_for(asyncio.run(exercise()))

    assert state["consecutive_failures"] == 1
    assert state["last_failure_type"] == type(error).__name__


def test_router_success_resets_previous_failure_counter() -> None:
    manager = ProviderHealthManager(failure_threshold=3)
    primary = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(primary, health_manager=manager)

    async def exercise() -> dict:
        await manager.record_failure("groq", LLMTimeoutError("private"))
        await router.chat(MESSAGES)
        return await manager.snapshot()

    state = state_for(asyncio.run(exercise()))

    assert state["consecutive_failures"] == 0


def test_router_authentication_stops_without_health_failure() -> None:
    manager = ProviderHealthManager(failure_threshold=1)
    primary = provider(
        "groq",
        error=LLMAuthenticationError("private"),
    )
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(primary, fallback, health_manager=manager)

    async def exercise() -> dict:
        with pytest.raises(LLMAuthenticationError):
            await router.chat(MESSAGES)
        return await manager.snapshot()

    snapshot = asyncio.run(exercise())

    assert snapshot == {}
    fallback.chat.assert_not_awaited()


def test_controlled_circuit_open_skip_cooldown_and_reset() -> None:
    clock = FakeClock()
    manager = ProviderHealthManager(
        failure_threshold=3,
        cooldown_seconds=60,
        clock=clock,
    )
    primary = provider("groq", error=LLMTimeoutError("private"))
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(primary, fallback, health_manager=manager)

    async def exercise() -> tuple[dict, dict]:
        for _ in range(3):
            assert await router.chat(MESSAGES) == OPENROUTER_RESULT

        primary.chat.reset_mock()
        fallback.chat.reset_mock()
        assert await router.chat(MESSAGES) == OPENROUTER_RESULT
        primary.chat.assert_not_awaited()
        fallback.chat.assert_awaited_once_with(MESSAGES)

        clock.advance(60)
        primary.chat.side_effect = None
        primary.chat.return_value = GROQ_RESULT
        primary.chat.reset_mock()
        fallback.chat.reset_mock()
        assert await router.chat(MESSAGES) == GROQ_RESULT
        primary.chat.assert_awaited_once_with(MESSAGES)
        fallback.chat.assert_not_awaited()
        return await manager.snapshot(), GROQ_RESULT

    snapshot, result = asyncio.run(exercise())

    assert result == GROQ_RESULT
    assert state_for(snapshot)["consecutive_failures"] == 0
    assert state_for(snapshot)["circuit_open"] is False


def test_disabled_circuit_preserves_router_behavior() -> None:
    manager = ProviderHealthManager(enabled=False, failure_threshold=1)
    primary = provider("groq", error=LLMTimeoutError("private"))
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(primary, fallback, health_manager=manager)

    async def exercise() -> None:
        assert await router.chat(MESSAGES) == OPENROUTER_RESULT
        assert await router.chat(MESSAGES) == OPENROUTER_RESULT

    asyncio.run(exercise())

    assert primary.chat.await_count == 2
    assert fallback.chat.await_count == 2


def test_disabled_provider_is_skipped_without_health_state() -> None:
    manager = ProviderHealthManager()
    primary = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(
        primary_provider=primary,
        fallback_provider=None,
        health_manager=manager,
    )

    async def exercise() -> dict:
        assert await router.chat(MESSAGES) == GROQ_RESULT
        return await manager.snapshot()

    snapshot = asyncio.run(exercise())

    assert snapshot == {}
