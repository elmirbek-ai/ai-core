import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.llm.budget import RequestBudgetPolicy
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMTimeoutError,
)
from app.llm.health import ProviderHealthManager
from app.llm.router import LLMRouter
from app.llm.task import TaskType


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
        close=AsyncMock(),
    )


def policy(
    *,
    enabled: bool = True,
    seconds: float = 1.0,
) -> RequestBudgetPolicy:
    return RequestBudgetPolicy(
        enabled=enabled,
        standard_seconds=seconds,
        reasoning_seconds=seconds,
        code_seconds=seconds,
        long_context_seconds=seconds,
        multimodal_seconds=seconds,
    )


def test_config_budget_defaults_are_valid() -> None:
    settings = Settings(_env_file=None, groq_api_key="test-key")

    assert settings.llm_request_budget_enabled is True
    assert settings.llm_standard_budget_seconds == 30.0
    assert settings.llm_reasoning_budget_seconds == 60.0
    assert settings.llm_code_budget_seconds == 60.0
    assert settings.llm_long_context_budget_seconds == 90.0
    assert settings.llm_multimodal_budget_seconds == 90.0


@pytest.mark.parametrize(
    "field",
    [
        "llm_standard_budget_seconds",
        "llm_reasoning_budget_seconds",
        "llm_code_budget_seconds",
        "llm_long_context_budget_seconds",
        "llm_multimodal_budget_seconds",
    ],
)
@pytest.mark.parametrize("invalid_value", [0, -1])
def test_zero_and_negative_config_budgets_are_invalid(
    field: str,
    invalid_value: int,
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            groq_api_key="test-key",
            **{field: invalid_value},
        )


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (TaskType.GENERAL, 30.0),
        (TaskType.FAST, 30.0),
        (TaskType.SUMMARIZE, 30.0),
        (TaskType.TRANSLATE, 30.0),
        (TaskType.CLASSIFY, 30.0),
        (TaskType.EXTRACT, 30.0),
        (TaskType.AUTO, 30.0),
        (TaskType.REASONING, 60.0),
        (TaskType.CODE, 61.0),
        (TaskType.LONG_CONTEXT, 90.0),
        (TaskType.MULTIMODAL, 91.0),
    ],
)
def test_task_budget_mapping(task: TaskType, expected: float) -> None:
    budget_policy = RequestBudgetPolicy(
        standard_seconds=30,
        reasoning_seconds=60,
        code_seconds=61,
        long_context_seconds=90,
        multimodal_seconds=91,
    )

    assert budget_policy.resolve(task) == expected


def test_disabled_policy_resolves_no_deadline() -> None:
    assert policy(enabled=False).resolve(TaskType.GENERAL) is None


def test_primary_success_within_budget() -> None:
    primary = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(primary, budget_policy=policy())

    result = asyncio.run(router.chat(MESSAGES))

    assert result == GROQ_RESULT


def test_recoverable_failure_then_fallback_succeeds_within_budget() -> None:
    primary = provider("groq", error=LLMTimeoutError("provider timeout"))
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(primary, fallback, budget_policy=policy())

    result = asyncio.run(router.chat(MESSAGES))

    assert result == OPENROUTER_RESULT


def test_total_budget_exhausted_before_next_fallback() -> None:
    clock = FakeClock()

    async def fail_after_consuming_budget(messages, model=None):
        clock.advance(1.0)
        raise LLMTimeoutError("provider timeout")

    primary = provider("groq")
    primary.chat.side_effect = fail_after_consuming_budget
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary,
        fallback,
        budget_policy=policy(seconds=1.0),
        clock=clock,
    )

    with pytest.raises(LLMTimeoutError, match="budget exhausted"):
        asyncio.run(router.chat(MESSAGES))

    fallback.chat.assert_not_awaited()


def test_open_circuit_skip_preserves_budget_for_fallback() -> None:
    clock = FakeClock()
    manager = ProviderHealthManager(
        failure_threshold=1,
        cooldown_seconds=60,
        clock=clock,
    )
    primary = provider("groq", result=GROQ_RESULT)
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary,
        fallback,
        health_manager=manager,
        budget_policy=policy(seconds=1.0),
        clock=clock,
    )

    async def exercise() -> dict[str, str]:
        await manager.record_failure("groq", LLMTimeoutError("private"))
        return await router.chat(MESSAGES)

    result = asyncio.run(exercise())

    assert result == OPENROUTER_RESULT
    primary.chat.assert_not_awaited()
    fallback.chat.assert_awaited_once_with(MESSAGES)


@pytest.mark.parametrize(
    "error",
    [LLMAuthenticationError("private"), LLMProviderError("private")],
)
def test_non_recoverable_errors_remain_terminal(error: Exception) -> None:
    primary = provider("groq", error=error)
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(primary, fallback, budget_policy=policy())

    with pytest.raises(type(error)):
        asyncio.run(router.chat(MESSAGES))

    fallback.chat.assert_not_awaited()


def test_hard_deadline_cancels_provider_records_health_and_stops_chain() -> None:
    cancelled = False

    async def hang(messages, model=None):
        nonlocal cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled = True
            raise

    manager = ProviderHealthManager(failure_threshold=3)
    primary = provider("groq")
    primary.chat.side_effect = hang
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary,
        fallback,
        health_manager=manager,
        budget_policy=policy(seconds=0.01),
    )

    async def exercise() -> dict:
        with pytest.raises(LLMTimeoutError, match="budget exhausted"):
            await router.chat(MESSAGES)
        return await manager.snapshot()

    snapshot = asyncio.run(exercise())

    assert cancelled is True
    assert snapshot["groq"]["consecutive_failures"] == 1
    assert snapshot["groq"]["last_failure_type"] == "LLMTimeoutError"
    fallback.chat.assert_not_awaited()
    primary.close.assert_not_awaited()


def test_timed_out_provider_remains_usable_on_next_request() -> None:
    attempt = 0

    async def first_hangs_then_succeeds(messages, model=None):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            await asyncio.Event().wait()
        return GROQ_RESULT

    manager = ProviderHealthManager(failure_threshold=3)
    primary = provider("groq")
    primary.chat.side_effect = first_hangs_then_succeeds
    router = LLMRouter(
        primary,
        health_manager=manager,
        budget_policy=policy(seconds=0.01),
    )

    async def exercise() -> tuple[dict[str, str], dict]:
        with pytest.raises(LLMTimeoutError):
            await router.chat(MESSAGES)
        result = await router.chat(MESSAGES)
        return result, await manager.snapshot()

    result, snapshot = asyncio.run(exercise())

    assert result == GROQ_RESULT
    assert snapshot["groq"]["consecutive_failures"] == 0
    assert primary.close.await_count == 0


def test_disabled_budget_preserves_pre_budget_fallback_behavior() -> None:
    clock = FakeClock()

    async def fail_after_time_passes(messages, model=None):
        clock.advance(100)
        raise LLMTimeoutError("provider timeout")

    primary = provider("groq")
    primary.chat.side_effect = fail_after_time_passes
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary,
        fallback,
        budget_policy=policy(enabled=False),
        clock=clock,
    )

    result = asyncio.run(router.chat(MESSAGES))

    assert result == OPENROUTER_RESULT


def test_concurrent_requests_have_independent_deadlines() -> None:
    slow_cancelled = False

    async def respond_by_message(messages, model=None):
        nonlocal slow_cancelled
        if messages[0]["content"] == "slow":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                slow_cancelled = True
                raise
        return GROQ_RESULT

    primary = provider("groq")
    primary.chat.side_effect = respond_by_message
    router = LLMRouter(
        primary,
        budget_policy=policy(seconds=0.01),
    )

    async def exercise():
        return await asyncio.gather(
            router.chat([{"role": "user", "content": "slow"}]),
            router.chat([{"role": "user", "content": "fast"}]),
            return_exceptions=True,
        )

    slow_result, fast_result = asyncio.run(exercise())

    assert isinstance(slow_result, LLMTimeoutError)
    assert fast_result == GROQ_RESULT
    assert slow_cancelled is True
