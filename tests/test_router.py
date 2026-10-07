import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
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
CLOUDFLARE_RESULT = {
    "provider": "cloudflare",
    "model": "cf/openai/gpt-oss-120b",
    "content": "cloudflare response",
}


def provider(name: str, *, result=None, error=None):
    chat = AsyncMock(return_value=result, side_effect=error)
    return SimpleNamespace(name=name, chat=chat)


def test_primary_success_does_not_call_fallback() -> None:
    primary = provider("groq", result=GROQ_RESULT)
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(primary, fallback)

    result = asyncio.run(router.chat(MESSAGES))

    assert result == GROQ_RESULT
    fallback.chat.assert_not_awaited()


@pytest.mark.parametrize(
    "error",
    [
        LLMRateLimitError("rate limited"),
        LLMTimeoutError("timed out"),
        LLMUpstreamError("upstream failed"),
    ],
)
def test_recoverable_primary_error_uses_fallback(error) -> None:
    primary = provider("groq", error=error)
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(primary, fallback)

    result = asyncio.run(router.chat(MESSAGES))

    assert result == OPENROUTER_RESULT
    assert result["provider"] == "openrouter"
    fallback.chat.assert_awaited_once_with(MESSAGES)


def test_authentication_error_does_not_call_fallback() -> None:
    primary = provider("groq", error=LLMAuthenticationError("auth failed"))
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(primary, fallback)

    with pytest.raises(LLMAuthenticationError):
        asyncio.run(router.chat(MESSAGES))

    fallback.chat.assert_not_awaited()


def test_disabled_fallback_propagates_original_error() -> None:
    original_error = LLMTimeoutError("timed out")
    router = LLMRouter(provider("groq", error=original_error))

    with pytest.raises(LLMTimeoutError) as captured:
        asyncio.run(router.chat(MESSAGES))

    assert captured.value is original_error


def test_fallback_failure_raises_base_provider_error() -> None:
    primary = provider("groq", error=LLMUpstreamError("primary failed"))
    fallback = provider(
        "openrouter",
        error=LLMTimeoutError("fallback failed"),
    )
    router = LLMRouter(primary, fallback)

    with pytest.raises(LLMProviderError) as captured:
        asyncio.run(router.chat(MESSAGES))

    assert type(captured.value) is LLMProviderError


def test_fallback_log_does_not_include_error_or_payload(
    caplog,
) -> None:
    secret = "sensitive-provider-detail"
    primary = provider("groq", error=LLMUpstreamError(secret))
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(primary, fallback)

    with caplog.at_level(logging.WARNING, logger="app.llm.router"):
        asyncio.run(router.chat(MESSAGES))

    assert "falling back to provider" in caplog.text
    assert secret not in caplog.text
    assert "Hello" not in caplog.text


def test_reasoning_model_is_used_only_for_primary_before_fallback() -> None:
    reasoning_model = "openai/gpt-oss-120b"
    primary = provider("groq", error=LLMTimeoutError("timed out"))
    fallback = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(primary, fallback)

    result = asyncio.run(
        router.chat(MESSAGES, model=reasoning_model),
    )

    assert result == OPENROUTER_RESULT
    primary.chat.assert_awaited_once_with(
        MESSAGES,
        model=reasoning_model,
    )
    fallback.chat.assert_awaited_once_with(MESSAGES)


@pytest.mark.parametrize(
    "task",
    [TaskType.MULTIMODAL, TaskType.LONG_CONTEXT],
)
def test_gemini_tasks_use_gemini_only_on_success(task) -> None:
    groq = provider("groq", result=GROQ_RESULT)
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    gemini_result = {
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "content": "gemini response",
    }
    gemini = provider("gemini", result=gemini_result)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        gemini_provider=gemini,
    )

    result = asyncio.run(router.chat(MESSAGES, task=task))

    assert result == gemini_result
    gemini.chat.assert_awaited_once_with(MESSAGES)
    openrouter.chat.assert_not_awaited()
    groq.chat.assert_not_awaited()


def test_gemini_recoverable_error_falls_back_to_openrouter() -> None:
    groq = provider("groq", result=GROQ_RESULT)
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    gemini = provider("gemini", error=LLMTimeoutError("timed out"))
    router = LLMRouter(groq, openrouter, gemini)

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.LONG_CONTEXT),
    )

    assert result == OPENROUTER_RESULT
    openrouter.chat.assert_awaited_once_with(MESSAGES)
    groq.chat.assert_not_awaited()


def test_gemini_failure_falls_back_to_groq_when_openrouter_disabled() -> None:
    groq = provider("groq", result=GROQ_RESULT)
    gemini = provider("gemini", error=LLMUpstreamError("failed"))
    router = LLMRouter(
        primary_provider=groq,
        gemini_provider=gemini,
    )

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.MULTIMODAL),
    )

    assert result == GROQ_RESULT
    groq.chat.assert_awaited_once_with(MESSAGES)


def test_gemini_authentication_error_does_not_fallback() -> None:
    groq = provider("groq", result=GROQ_RESULT)
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    gemini = provider(
        "gemini",
        error=LLMAuthenticationError("auth failed"),
    )
    router = LLMRouter(groq, openrouter, gemini)

    with pytest.raises(LLMAuthenticationError):
        asyncio.run(
            router.chat(MESSAGES, task=TaskType.LONG_CONTEXT),
        )

    openrouter.chat.assert_not_awaited()
    groq.chat.assert_not_awaited()


def test_general_task_keeps_existing_groq_flow_with_gemini_enabled() -> None:
    groq = provider("groq", result=GROQ_RESULT)
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    gemini = provider(
        "gemini",
        result={
            "provider": "gemini",
            "model": "gemini-3.8-flash",
            "content": "gemini response",
        },
    )
    router = LLMRouter(groq, openrouter, gemini)

    result = asyncio.run(
        router.chat(
            MESSAGES,
            model="openai/gpt-oss-20b",
            task=TaskType.GENERAL,
        )
    )

    assert result == GROQ_RESULT
    groq.chat.assert_awaited_once_with(
        MESSAGES,
        model="openai/gpt-oss-20b",
    )
    gemini.chat.assert_not_awaited()


def test_fast_chain_groq_success_calls_only_groq() -> None:
    groq = provider("groq", result=GROQ_RESULT)
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    cloudflare = provider("cloudflare", result=CLOUDFLARE_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
    )

    result = asyncio.run(
        router.chat(
            MESSAGES,
            model="openai/gpt-oss-20b",
            task=TaskType.GENERAL,
        )
    )

    assert result == GROQ_RESULT
    openrouter.chat.assert_not_awaited()
    cloudflare.chat.assert_not_awaited()


def test_fast_chain_uses_openrouter_before_cloudflare() -> None:
    groq = provider("groq", error=LLMTimeoutError("timed out"))
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    cloudflare = provider("cloudflare", result=CLOUDFLARE_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
    )

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.GENERAL),
    )

    assert result == OPENROUTER_RESULT
    cloudflare.chat.assert_not_awaited()


def test_fast_chain_uses_cloudflare_after_openrouter_failure() -> None:
    groq = provider("groq", error=LLMTimeoutError("groq failed"))
    openrouter = provider(
        "openrouter",
        error=LLMUpstreamError("openrouter failed"),
    )
    cloudflare = provider("cloudflare", result=CLOUDFLARE_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
    )

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.GENERAL),
    )

    assert result == CLOUDFLARE_RESULT
    cloudflare.chat.assert_awaited_once_with(MESSAGES)


def test_reasoning_chain_groq_success_calls_only_groq() -> None:
    groq = provider("groq", result=GROQ_RESULT)
    cloudflare = provider("cloudflare", result=CLOUDFLARE_RESULT)
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
    )

    result = asyncio.run(
        router.chat(
            MESSAGES,
            model="openai/gpt-oss-120b",
            task=TaskType.REASONING,
        )
    )

    assert result == GROQ_RESULT
    cloudflare.chat.assert_not_awaited()
    openrouter.chat.assert_not_awaited()


def test_reasoning_chain_uses_cloudflare_before_openrouter() -> None:
    groq = provider("groq", error=LLMTimeoutError("timed out"))
    cloudflare = provider("cloudflare", result=CLOUDFLARE_RESULT)
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
    )

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.REASONING),
    )

    assert result == CLOUDFLARE_RESULT
    openrouter.chat.assert_not_awaited()


def test_reasoning_chain_uses_openrouter_after_cloudflare_failure() -> None:
    groq = provider("groq", error=LLMTimeoutError("groq failed"))
    cloudflare = provider(
        "cloudflare",
        error=LLMUpstreamError("cloudflare failed"),
    )
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
    )

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.CODE),
    )

    assert result == OPENROUTER_RESULT
    openrouter.chat.assert_awaited_once_with(MESSAGES)


def test_long_context_uses_cloudflare_after_two_failures() -> None:
    gemini = provider("gemini", error=LLMTimeoutError("gemini failed"))
    openrouter = provider(
        "openrouter",
        error=LLMUpstreamError("openrouter failed"),
    )
    cloudflare = provider("cloudflare", result=CLOUDFLARE_RESULT)
    groq = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        gemini_provider=gemini,
        cloudflare_provider=cloudflare,
    )

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.LONG_CONTEXT),
    )

    assert result == CLOUDFLARE_RESULT
    groq.chat.assert_not_awaited()


def test_long_context_uses_groq_after_three_failures() -> None:
    gemini = provider("gemini", error=LLMTimeoutError("gemini failed"))
    openrouter = provider(
        "openrouter",
        error=LLMUpstreamError("openrouter failed"),
    )
    cloudflare = provider(
        "cloudflare",
        error=LLMRateLimitError("cloudflare failed"),
    )
    groq = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        gemini_provider=gemini,
        cloudflare_provider=cloudflare,
    )

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.LONG_CONTEXT),
    )

    assert result == GROQ_RESULT
    groq.chat.assert_awaited_once_with(MESSAGES)


def test_cloudflare_authentication_error_stops_reasoning_chain() -> None:
    groq = provider("groq", error=LLMTimeoutError("groq failed"))
    cloudflare = provider(
        "cloudflare",
        error=LLMAuthenticationError("auth failed"),
    )
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
    )

    with pytest.raises(LLMAuthenticationError):
        asyncio.run(
            router.chat(MESSAGES, task=TaskType.REASONING),
        )

    openrouter.chat.assert_not_awaited()


def test_multimodal_chain_does_not_use_cloudflare() -> None:
    gemini = provider("gemini", error=LLMTimeoutError("gemini failed"))
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    cloudflare = provider("cloudflare", result=CLOUDFLARE_RESULT)
    groq = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        gemini_provider=gemini,
        cloudflare_provider=cloudflare,
    )

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.MULTIMODAL),
    )

    assert result == OPENROUTER_RESULT
    cloudflare.chat.assert_not_awaited()
