import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.llm.budget import RequestBudgetPolicy
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.health import ProviderHealthManager
from app.llm.router import LLMRouter
from app.llm.task import TaskType
from app.llm.telemetry import LLMTelemetry


MESSAGES = [{"role": "user", "content": "private prompt"}]
GROQ_RESULT = {
    "provider": "groq",
    "model": "openai/gpt-oss-20b",
    "content": "private response",
}
OPENROUTER_RESULT = {
    "provider": "openrouter",
    "model": "openrouter/free",
    "content": "fallback response",
}
CLOUDFLARE_RESULT = {
    "provider": "cloudflare",
    "model": "cf/openai/gpt-oss-120b",
    "content": "cloudflare response",
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


def short_budget(seconds: float = 1.0) -> RequestBudgetPolicy:
    return RequestBudgetPolicy(
        standard_seconds=seconds,
        reasoning_seconds=seconds,
        code_seconds=seconds,
        long_context_seconds=seconds,
        multimodal_seconds=seconds,
    )


def test_config_telemetry_enabled_by_default_and_can_be_disabled() -> None:
    enabled = Settings(_env_file=None, groq_api_key="test-key")
    disabled = Settings(
        _env_file=None,
        groq_api_key="test-key",
        llm_telemetry_enabled=False,
    )

    assert enabled.llm_telemetry_enabled is True
    assert disabled.llm_telemetry_enabled is False


def test_first_provider_success_records_metrics() -> None:
    clock = FakeClock()
    telemetry = LLMTelemetry(clock=clock)

    async def exercise() -> dict:
        await telemetry.record_provider_success("groq", 0.25)
        return await telemetry.snapshot()

    metrics = asyncio.run(exercise())["providers"]["groq"]

    assert metrics["attempts"] == 1
    assert metrics["successes"] == 1
    assert metrics["total_latency_seconds"] == 0.25
    assert metrics["last_latency_seconds"] == 0.25
    assert metrics["average_latency_seconds"] == 0.25
    assert metrics["last_success_time"] == 100.0


def test_multiple_successes_compute_average_latency() -> None:
    telemetry = LLMTelemetry()

    async def exercise() -> dict:
        await telemetry.record_provider_success("groq", 0.2)
        await telemetry.record_provider_success("groq", 0.4)
        return await telemetry.snapshot()

    metrics = asyncio.run(exercise())["providers"]["groq"]

    assert metrics["attempts"] == 2
    assert metrics["successes"] == 2
    assert metrics["total_latency_seconds"] == pytest.approx(0.6)
    assert metrics["average_latency_seconds"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("error", "specific_counter", "recoverable"),
    [
        (LLMTimeoutError("private"), "timeout_failures", True),
        (LLMRateLimitError("private"), "rate_limit_failures", True),
        (LLMUpstreamError("private"), "upstream_failures", True),
        (
            LLMAuthenticationError("private"),
            "authentication_failures",
            False,
        ),
        (LLMProviderError("private"), "provider_failures", False),
    ],
)
def test_provider_failure_categories(
    error: Exception,
    specific_counter: str,
    recoverable: bool,
) -> None:
    telemetry = LLMTelemetry()

    async def exercise() -> dict:
        await telemetry.record_provider_failure("groq", error, 0.5)
        return await telemetry.snapshot()

    metrics = asyncio.run(exercise())["providers"]["groq"]

    assert metrics["attempts"] == 1
    assert metrics[specific_counter] == 1
    assert metrics["recoverable_failures"] == int(recoverable)
    assert metrics["last_failure_time"] is not None


def test_different_provider_metrics_are_independent() -> None:
    telemetry = LLMTelemetry()

    async def exercise() -> dict:
        await telemetry.record_provider_success("groq", 0.1)
        await telemetry.record_provider_failure(
            "openrouter",
            LLMTimeoutError("private"),
            0.2,
        )
        return await telemetry.snapshot()

    providers = asyncio.run(exercise())["providers"]

    assert providers["groq"]["successes"] == 1
    assert providers["groq"]["timeout_failures"] == 0
    assert providers["openrouter"]["successes"] == 0
    assert providers["openrouter"]["timeout_failures"] == 1


def test_primary_success_records_request_depth_zero() -> None:
    clock = FakeClock()

    async def succeed(messages, model=None):
        clock.advance(0.2)
        return GROQ_RESULT

    telemetry = LLMTelemetry(clock=clock)
    primary = provider("groq")
    primary.chat.side_effect = succeed
    router = LLMRouter(primary, telemetry=telemetry, clock=clock)

    result = asyncio.run(router.chat(MESSAGES))
    snapshot = asyncio.run(telemetry.snapshot())

    assert result == GROQ_RESULT
    assert snapshot["providers"]["groq"]["attempts"] == 1
    assert snapshot["providers"]["groq"]["successes"] == 1
    assert snapshot["requests"]["total"] == 1
    assert snapshot["requests"]["successes"] == 1
    assert snapshot["requests"]["last_fallback_depth"] == 0
    assert snapshot["requests"]["last_selected_provider"] == "groq"
    assert snapshot["requests"]["last_task"] == "general"
    assert snapshot["requests"]["total_latency_seconds"] == pytest.approx(0.2)


def test_fallback_success_records_depth_one() -> None:
    telemetry = LLMTelemetry()
    primary = provider("groq", error=LLMTimeoutError("private"))
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(primary, fallback, telemetry=telemetry)

    result = asyncio.run(router.chat(MESSAGES))
    snapshot = asyncio.run(telemetry.snapshot())

    assert result == OPENROUTER_RESULT
    assert snapshot["providers"]["groq"]["timeout_failures"] == 1
    assert snapshot["providers"]["openrouter"]["successes"] == 1
    assert snapshot["requests"]["last_fallback_depth"] == 1
    assert snapshot["requests"]["last_selected_provider"] == "openrouter"


def test_deeper_fallback_records_actual_attempted_failures() -> None:
    telemetry = LLMTelemetry()
    primary = provider("groq", error=LLMTimeoutError("private"))
    fallback = provider("openrouter", error=LLMUpstreamError("private"))
    cloudflare = provider("cloudflare", result=CLOUDFLARE_RESULT)
    router = LLMRouter(
        primary_provider=primary,
        fallback_provider=fallback,
        cloudflare_provider=cloudflare,
        telemetry=telemetry,
    )

    result = asyncio.run(router.chat(MESSAGES))
    snapshot = asyncio.run(telemetry.snapshot())

    assert result == CLOUDFLARE_RESULT
    assert snapshot["requests"]["last_fallback_depth"] == 2
    assert snapshot["requests"]["last_selected_provider"] == "cloudflare"


def test_terminal_failure_records_request_failure() -> None:
    telemetry = LLMTelemetry()
    primary = provider("groq", error=LLMProviderError("private raw error"))
    router = LLMRouter(primary, telemetry=telemetry)

    with pytest.raises(LLMProviderError):
        asyncio.run(router.chat(MESSAGES))
    snapshot = asyncio.run(telemetry.snapshot())

    assert snapshot["providers"]["groq"]["provider_failures"] == 1
    assert snapshot["requests"]["total"] == 1
    assert snapshot["requests"]["failures"] == 1
    assert snapshot["requests"]["last_error_category"] == "provider"
    assert snapshot["requests"]["last_selected_provider"] is None


def test_circuit_skip_is_not_an_attempt_or_fallback_depth() -> None:
    manager = ProviderHealthManager(failure_threshold=1)
    telemetry = LLMTelemetry()
    primary = provider("groq", result=GROQ_RESULT)
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary,
        fallback,
        health_manager=manager,
        telemetry=telemetry,
    )

    async def exercise() -> tuple[dict[str, str], dict]:
        await manager.record_failure("groq", LLMTimeoutError("private"))
        result = await router.chat(MESSAGES)
        return result, await telemetry.snapshot()

    result, snapshot = asyncio.run(exercise())

    assert result == OPENROUTER_RESULT
    assert snapshot["providers"]["groq"]["attempts"] == 0
    assert snapshot["providers"]["groq"]["circuit_skips"] == 1
    assert snapshot["requests"]["last_fallback_depth"] == 0
    primary.chat.assert_not_awaited()


def test_request_deadline_records_provider_timeout_and_request_failure() -> None:
    async def hang(messages, model=None):
        await asyncio.Event().wait()

    telemetry = LLMTelemetry()
    manager = ProviderHealthManager(failure_threshold=3)
    primary = provider("groq")
    primary.chat.side_effect = hang
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary,
        fallback,
        health_manager=manager,
        budget_policy=short_budget(0.01),
        telemetry=telemetry,
    )

    with pytest.raises(LLMTimeoutError):
        asyncio.run(router.chat(MESSAGES))
    snapshot = asyncio.run(telemetry.snapshot())

    assert snapshot["providers"]["groq"]["attempts"] == 1
    assert snapshot["providers"]["groq"]["timeout_failures"] == 1
    assert snapshot["requests"]["failures"] == 1
    assert snapshot["requests"]["last_error_category"] == "timeout"
    fallback.chat.assert_not_awaited()


def test_request_aggregates_total_and_average_latency() -> None:
    telemetry = LLMTelemetry()

    async def exercise() -> dict:
        await telemetry.record_request_success(
            TaskType.GENERAL,
            "groq",
            0.2,
            0,
        )
        await telemetry.record_request_failure(
            TaskType.GENERAL,
            LLMTimeoutError("private"),
            0.4,
            1,
        )
        return await telemetry.snapshot()

    requests = asyncio.run(exercise())["requests"]

    assert requests["total"] == 2
    assert requests["successes"] == 1
    assert requests["failures"] == 1
    assert requests["total_latency_seconds"] == pytest.approx(0.6)
    assert requests["average_latency_seconds"] == pytest.approx(0.3)
    assert requests["average_fallback_depth"] == pytest.approx(0.5)
    assert requests["by_task"]["general"]["total"] == 2


def test_disabled_telemetry_does_not_mutate_or_change_routing() -> None:
    telemetry = LLMTelemetry(enabled=False)
    primary = provider("groq", error=LLMTimeoutError("private"))
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(primary, fallback, telemetry=telemetry)

    result = asyncio.run(router.chat(MESSAGES))
    snapshot = asyncio.run(telemetry.snapshot())

    assert result == OPENROUTER_RESULT
    assert snapshot["providers"] == {}
    assert snapshot["requests"]["total"] == 0


def test_concurrent_updates_do_not_lose_counters() -> None:
    telemetry = LLMTelemetry()

    async def exercise() -> dict:
        await asyncio.gather(
            *(
                telemetry.record_provider_success("groq", 0.01)
                for _ in range(100)
            )
        )
        return await telemetry.snapshot()

    metrics = asyncio.run(exercise())["providers"]["groq"]

    assert metrics["attempts"] == 100
    assert metrics["successes"] == 100
    assert metrics["total_latency_seconds"] == pytest.approx(1.0)


def test_snapshot_is_deterministic_and_not_a_mutable_internal_leak() -> None:
    telemetry = LLMTelemetry()

    async def exercise() -> tuple[dict, dict]:
        await telemetry.record_provider_success("groq", 0.1)
        first = await telemetry.snapshot()
        first["providers"]["groq"]["attempts"] = 999
        first["requests"]["by_task"]["injected"] = {"total": 999}
        return first, await telemetry.snapshot()

    _, second = asyncio.run(exercise())

    assert second["providers"]["groq"]["attempts"] == 1
    assert "injected" not in second["requests"]["by_task"]
    assert set(second) == {"providers", "requests", "streaming"}


def test_snapshot_contains_no_payload_or_raw_error_secrets() -> None:
    telemetry = LLMTelemetry()
    secrets = (
        "private prompt",
        "private response",
        "secret-api-key",
        "raw-sdk-error",
        "private-account-id",
    )

    async def exercise() -> dict:
        await telemetry.record_provider_failure(
            "groq",
            LLMProviderError("raw-sdk-error secret-api-key"),
            0.1,
        )
        await telemetry.record_request_failure(
            TaskType.GENERAL,
            LLMProviderError("private-account-id"),
            0.1,
            0,
        )
        return await telemetry.snapshot()

    serialized = json.dumps(asyncio.run(exercise()))

    for secret in secrets:
        assert secret not in serialized
