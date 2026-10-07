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
OLLAMA_RESULT = {
    "provider": "ollama",
    "model": "gpt-oss:120b",
    "content": "ollama response",
}
KILO_RESULT = {
    "provider": "kilo",
    "model": "stepfun/step-3.7-flash",
    "content": "kilo response",
}
KILO_GENERAL_MODEL = "stepfun/step-3.7-flash:free"
KILO_CODE_MODEL = "cohere/north-mini-code:free"
KILO_LONG_MODEL = "dots-studio/dots-3-note-preview:free"
LLM7_RESULT = {
    "provider": "llm7",
    "model": "gpt-oss:20b",
    "content": "llm7 response",
}
LLM7_GENERAL_MODEL = "gpt-oss:20b"
LLM7_REASONING_MODEL = "gpt-oss:20b"
LLM7_CODE_MODEL = "gpt-oss:20b"


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


def test_fast_chain_uses_ollama_after_three_failures() -> None:
    groq = provider("groq", error=LLMTimeoutError("groq failed"))
    openrouter = provider(
        "openrouter",
        error=LLMUpstreamError("openrouter failed"),
    )
    cloudflare = provider(
        "cloudflare",
        error=LLMRateLimitError("cloudflare failed"),
    )
    ollama = provider("ollama", result=OLLAMA_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
        ollama_provider=ollama,
    )

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.GENERAL),
    )

    assert result == OLLAMA_RESULT
    ollama.chat.assert_awaited_once_with(MESSAGES)


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


def test_reasoning_chain_uses_ollama_before_openrouter() -> None:
    groq = provider("groq", error=LLMTimeoutError("groq failed"))
    cloudflare = provider(
        "cloudflare",
        error=LLMUpstreamError("cloudflare failed"),
    )
    ollama = provider("ollama", result=OLLAMA_RESULT)
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
        ollama_provider=ollama,
    )

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.REASONING),
    )

    assert result == OLLAMA_RESULT
    ollama.chat.assert_awaited_once_with(MESSAGES)
    openrouter.chat.assert_not_awaited()


def test_reasoning_chain_continues_after_ollama_recoverable_error() -> None:
    groq = provider("groq", error=LLMTimeoutError("groq failed"))
    cloudflare = provider(
        "cloudflare",
        error=LLMUpstreamError("cloudflare failed"),
    )
    ollama = provider("ollama", error=LLMRateLimitError("ollama failed"))
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
        ollama_provider=ollama,
    )

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.CODE),
    )

    assert result == OPENROUTER_RESULT
    ollama.chat.assert_awaited_once_with(MESSAGES)
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


def test_long_context_uses_ollama_before_groq() -> None:
    gemini = provider("gemini", error=LLMTimeoutError("gemini failed"))
    openrouter = provider(
        "openrouter",
        error=LLMUpstreamError("openrouter failed"),
    )
    cloudflare = provider(
        "cloudflare",
        error=LLMRateLimitError("cloudflare failed"),
    )
    ollama = provider("ollama", result=OLLAMA_RESULT)
    groq = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        gemini_provider=gemini,
        cloudflare_provider=cloudflare,
        ollama_provider=ollama,
    )

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.LONG_CONTEXT),
    )

    assert result == OLLAMA_RESULT
    ollama.chat.assert_awaited_once_with(MESSAGES)
    groq.chat.assert_not_awaited()


def test_long_context_continues_after_ollama_recoverable_error() -> None:
    gemini = provider("gemini", error=LLMTimeoutError("gemini failed"))
    openrouter = provider(
        "openrouter",
        error=LLMUpstreamError("openrouter failed"),
    )
    cloudflare = provider(
        "cloudflare",
        error=LLMRateLimitError("cloudflare failed"),
    )
    ollama = provider("ollama", error=LLMTimeoutError("ollama failed"))
    groq = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        gemini_provider=gemini,
        cloudflare_provider=cloudflare,
        ollama_provider=ollama,
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


def test_multimodal_chain_does_not_use_ollama() -> None:
    gemini = provider("gemini", error=LLMTimeoutError("gemini failed"))
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    ollama = provider("ollama", result=OLLAMA_RESULT)
    groq = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        gemini_provider=gemini,
        ollama_provider=ollama,
    )

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.MULTIMODAL),
    )

    assert result == OPENROUTER_RESULT
    ollama.chat.assert_not_awaited()


def test_ollama_authentication_error_stops_reasoning_chain() -> None:
    groq = provider("groq", error=LLMTimeoutError("groq failed"))
    cloudflare = provider(
        "cloudflare",
        error=LLMUpstreamError("cloudflare failed"),
    )
    ollama = provider(
        "ollama",
        error=LLMAuthenticationError("auth failed"),
    )
    openrouter = provider("openrouter", result=OPENROUTER_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
        ollama_provider=ollama,
    )

    with pytest.raises(LLMAuthenticationError):
        asyncio.run(
            router.chat(MESSAGES, task=TaskType.REASONING),
        )

    openrouter.chat.assert_not_awaited()


def test_duplicate_provider_instance_is_called_only_once() -> None:
    groq = provider("groq", error=LLMTimeoutError("failed"))
    cloudflare = provider("cloudflare", result=CLOUDFLARE_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=groq,
        cloudflare_provider=cloudflare,
    )

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.GENERAL),
    )

    assert result == CLOUDFLARE_RESULT
    groq.chat.assert_awaited_once()


def test_long_context_skips_all_disabled_optional_providers() -> None:
    groq = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(primary_provider=groq)

    result = asyncio.run(
        router.chat(MESSAGES, task=TaskType.LONG_CONTEXT),
    )

    assert result == GROQ_RESULT
    groq.chat.assert_awaited_once_with(MESSAGES)


def test_openrouter_authentication_error_stops_fast_chain() -> None:
    groq = provider("groq", error=LLMTimeoutError("groq failed"))
    openrouter = provider(
        "openrouter",
        error=LLMAuthenticationError("auth failed"),
    )
    cloudflare = provider("cloudflare", result=CLOUDFLARE_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
    )

    with pytest.raises(LLMAuthenticationError):
        asyncio.run(
            router.chat(MESSAGES, task=TaskType.GENERAL),
        )

    cloudflare.chat.assert_not_awaited()


@pytest.mark.parametrize("task", list(TaskType))
def test_every_task_type_has_a_provider_route(task: TaskType) -> None:
    groq = provider("groq", result=GROQ_RESULT)
    gemini = provider(
        "gemini",
        result={
            "provider": "gemini",
            "model": "gemini-3.8-flash",
            "content": "gemini response",
        },
    )
    router = LLMRouter(
        primary_provider=groq,
        gemini_provider=gemini,
    )

    result = asyncio.run(router.chat(MESSAGES, task=task))

    if task in {TaskType.MULTIMODAL, TaskType.LONG_CONTEXT}:
        assert result["provider"] == "gemini"
    else:
        assert result["provider"] == "groq"


def test_standard_chain_reaches_kilo_last_with_general_model() -> None:
    groq = provider("groq", error=LLMTimeoutError("failed"))
    openrouter = provider("openrouter", error=LLMUpstreamError("failed"))
    cloudflare = provider("cloudflare", error=LLMRateLimitError("failed"))
    ollama = provider("ollama", error=LLMTimeoutError("failed"))
    kilo = provider("kilo", result=KILO_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
        ollama_provider=ollama,
        kilo_provider=kilo,
        kilo_general_model=KILO_GENERAL_MODEL,
    )

    result = asyncio.run(router.chat(MESSAGES, task=TaskType.GENERAL))

    assert result == KILO_RESULT
    kilo.chat.assert_awaited_once_with(MESSAGES, model=KILO_GENERAL_MODEL)


def test_code_chain_reaches_kilo_last_with_code_model() -> None:
    groq = provider("groq", error=LLMTimeoutError("failed"))
    cloudflare = provider("cloudflare", error=LLMUpstreamError("failed"))
    ollama = provider("ollama", error=LLMRateLimitError("failed"))
    openrouter = provider("openrouter", error=LLMTimeoutError("failed"))
    kilo_result = {**KILO_RESULT, "model": KILO_CODE_MODEL}
    kilo = provider("kilo", result=kilo_result)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
        ollama_provider=ollama,
        kilo_provider=kilo,
        kilo_code_model=KILO_CODE_MODEL,
    )

    result = asyncio.run(router.chat(MESSAGES, task=TaskType.CODE))

    assert result == kilo_result
    kilo.chat.assert_awaited_once_with(MESSAGES, model=KILO_CODE_MODEL)


def test_long_context_uses_kilo_before_cloudflare_with_long_model() -> None:
    gemini = provider("gemini", error=LLMTimeoutError("failed"))
    openrouter = provider("openrouter", error=LLMUpstreamError("failed"))
    kilo_result = {**KILO_RESULT, "model": KILO_LONG_MODEL}
    kilo = provider("kilo", result=kilo_result)
    cloudflare = provider("cloudflare", result=CLOUDFLARE_RESULT)
    ollama = provider("ollama", result=OLLAMA_RESULT)
    groq = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        gemini_provider=gemini,
        cloudflare_provider=cloudflare,
        ollama_provider=ollama,
        kilo_provider=kilo,
        kilo_long_context_model=KILO_LONG_MODEL,
    )

    result = asyncio.run(router.chat(MESSAGES, task=TaskType.LONG_CONTEXT))

    assert result == kilo_result
    kilo.chat.assert_awaited_once_with(MESSAGES, model=KILO_LONG_MODEL)
    cloudflare.chat.assert_not_awaited()
    ollama.chat.assert_not_awaited()
    groq.chat.assert_not_awaited()


def test_long_context_continues_after_kilo_recoverable_error() -> None:
    gemini = provider("gemini", error=LLMTimeoutError("failed"))
    openrouter = provider("openrouter", error=LLMUpstreamError("failed"))
    kilo = provider("kilo", error=LLMRateLimitError("failed"))
    cloudflare = provider("cloudflare", result=CLOUDFLARE_RESULT)
    groq = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        gemini_provider=gemini,
        cloudflare_provider=cloudflare,
        kilo_provider=kilo,
        kilo_long_context_model=KILO_LONG_MODEL,
    )

    result = asyncio.run(router.chat(MESSAGES, task=TaskType.LONG_CONTEXT))

    assert result == CLOUDFLARE_RESULT
    cloudflare.chat.assert_awaited_once_with(MESSAGES)
    groq.chat.assert_not_awaited()


def test_kilo_authentication_error_stops_long_context_chain() -> None:
    gemini = provider("gemini", error=LLMTimeoutError("failed"))
    openrouter = provider("openrouter", error=LLMUpstreamError("failed"))
    kilo = provider("kilo", error=LLMAuthenticationError("failed"))
    cloudflare = provider("cloudflare", result=CLOUDFLARE_RESULT)
    groq = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        gemini_provider=gemini,
        cloudflare_provider=cloudflare,
        kilo_provider=kilo,
        kilo_long_context_model=KILO_LONG_MODEL,
    )

    with pytest.raises(LLMAuthenticationError):
        asyncio.run(router.chat(MESSAGES, task=TaskType.LONG_CONTEXT))

    cloudflare.chat.assert_not_awaited()
    groq.chat.assert_not_awaited()


def test_kilo_non_recoverable_error_stops_long_context_chain() -> None:
    gemini = provider("gemini", error=LLMTimeoutError("failed"))
    openrouter = provider("openrouter", error=LLMUpstreamError("failed"))
    kilo = provider("kilo", error=LLMProviderError("failed"))
    cloudflare = provider("cloudflare", result=CLOUDFLARE_RESULT)
    groq = provider("groq", result=GROQ_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        gemini_provider=gemini,
        cloudflare_provider=cloudflare,
        kilo_provider=kilo,
        kilo_long_context_model=KILO_LONG_MODEL,
    )

    with pytest.raises(LLMProviderError):
        asyncio.run(router.chat(MESSAGES, task=TaskType.LONG_CONTEXT))

    cloudflare.chat.assert_not_awaited()
    groq.chat.assert_not_awaited()


def test_reasoning_and_multimodal_never_use_kilo() -> None:
    kilo = provider("kilo", result=KILO_RESULT)

    reasoning_groq = provider("groq", result=GROQ_RESULT)
    reasoning_router = LLMRouter(
        primary_provider=reasoning_groq,
        kilo_provider=kilo,
    )
    asyncio.run(reasoning_router.chat(MESSAGES, task=TaskType.REASONING))

    multimodal_groq = provider("groq", result=GROQ_RESULT)
    multimodal_router = LLMRouter(
        primary_provider=multimodal_groq,
        kilo_provider=kilo,
    )
    asyncio.run(multimodal_router.chat(MESSAGES, task=TaskType.MULTIMODAL))

    kilo.chat.assert_not_awaited()


def test_disabled_kilo_is_skipped_in_standard_chain() -> None:
    groq = provider("groq", error=LLMTimeoutError("failed"))
    openrouter = provider("openrouter", error=LLMUpstreamError("failed"))
    cloudflare = provider("cloudflare", error=LLMRateLimitError("failed"))
    ollama = provider("ollama", result=OLLAMA_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
        ollama_provider=ollama,
        kilo_provider=None,
    )

    result = asyncio.run(router.chat(MESSAGES, task=TaskType.GENERAL))

    assert result == OLLAMA_RESULT


def test_standard_chain_reaches_llm7_after_kilo() -> None:
    groq = provider("groq", error=LLMTimeoutError("failed"))
    openrouter = provider("openrouter", error=LLMUpstreamError("failed"))
    cloudflare = provider("cloudflare", error=LLMRateLimitError("failed"))
    ollama = provider("ollama", error=LLMTimeoutError("failed"))
    kilo = provider("kilo", error=LLMUpstreamError("failed"))
    llm7 = provider("llm7", result=LLM7_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
        ollama_provider=ollama,
        kilo_provider=kilo,
        kilo_general_model=KILO_GENERAL_MODEL,
        llm7_provider=llm7,
        llm7_general_model=LLM7_GENERAL_MODEL,
    )

    result = asyncio.run(router.chat(MESSAGES, task=TaskType.GENERAL))

    assert result == LLM7_RESULT
    kilo.chat.assert_awaited_once_with(MESSAGES, model=KILO_GENERAL_MODEL)
    llm7.chat.assert_awaited_once_with(MESSAGES, model=LLM7_GENERAL_MODEL)


def test_reasoning_chain_reaches_llm7_last() -> None:
    groq = provider("groq", error=LLMTimeoutError("failed"))
    cloudflare = provider("cloudflare", error=LLMUpstreamError("failed"))
    ollama = provider("ollama", error=LLMRateLimitError("failed"))
    openrouter = provider("openrouter", error=LLMTimeoutError("failed"))
    llm7_result = {**LLM7_RESULT, "model": LLM7_REASONING_MODEL}
    llm7 = provider("llm7", result=llm7_result)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
        ollama_provider=ollama,
        llm7_provider=llm7,
        llm7_reasoning_model=LLM7_REASONING_MODEL,
    )

    result = asyncio.run(router.chat(MESSAGES, task=TaskType.REASONING))

    assert result == llm7_result
    llm7.chat.assert_awaited_once_with(
        MESSAGES,
        model=LLM7_REASONING_MODEL,
    )


def test_code_chain_reaches_llm7_after_kilo() -> None:
    groq = provider("groq", error=LLMTimeoutError("failed"))
    cloudflare = provider("cloudflare", error=LLMUpstreamError("failed"))
    ollama = provider("ollama", error=LLMRateLimitError("failed"))
    openrouter = provider("openrouter", error=LLMTimeoutError("failed"))
    kilo = provider("kilo", error=LLMUpstreamError("failed"))
    llm7_result = {**LLM7_RESULT, "model": LLM7_CODE_MODEL}
    llm7 = provider("llm7", result=llm7_result)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
        ollama_provider=ollama,
        kilo_provider=kilo,
        kilo_code_model=KILO_CODE_MODEL,
        llm7_provider=llm7,
        llm7_code_model=LLM7_CODE_MODEL,
    )

    result = asyncio.run(router.chat(MESSAGES, task=TaskType.CODE))

    assert result == llm7_result
    kilo.chat.assert_awaited_once_with(MESSAGES, model=KILO_CODE_MODEL)
    llm7.chat.assert_awaited_once_with(MESSAGES, model=LLM7_CODE_MODEL)


@pytest.mark.parametrize(
    "task",
    [TaskType.LONG_CONTEXT, TaskType.MULTIMODAL],
)
def test_llm7_is_not_used_for_unsupported_tasks(task: TaskType) -> None:
    groq = provider("groq", result=GROQ_RESULT)
    llm7 = provider("llm7", result=LLM7_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        llm7_provider=llm7,
        llm7_general_model=LLM7_GENERAL_MODEL,
        llm7_reasoning_model=LLM7_REASONING_MODEL,
        llm7_code_model=LLM7_CODE_MODEL,
    )

    result = asyncio.run(router.chat(MESSAGES, task=task))

    assert result == GROQ_RESULT
    llm7.chat.assert_not_awaited()


def test_disabled_llm7_is_skipped_without_changing_kilo_result() -> None:
    groq = provider("groq", error=LLMTimeoutError("failed"))
    openrouter = provider("openrouter", error=LLMUpstreamError("failed"))
    cloudflare = provider("cloudflare", error=LLMRateLimitError("failed"))
    ollama = provider("ollama", error=LLMTimeoutError("failed"))
    kilo = provider("kilo", result=KILO_RESULT)
    router = LLMRouter(
        primary_provider=groq,
        fallback_provider=openrouter,
        cloudflare_provider=cloudflare,
        ollama_provider=ollama,
        kilo_provider=kilo,
        kilo_general_model=KILO_GENERAL_MODEL,
        llm7_provider=None,
    )

    result = asyncio.run(router.chat(MESSAGES, task=TaskType.GENERAL))

    assert result == KILO_RESULT


def test_llm7_recoverable_failure_uses_existing_terminal_semantics() -> None:
    groq = provider("groq", error=LLMTimeoutError("failed"))
    llm7 = provider("llm7", error=LLMUpstreamError("failed"))
    router = LLMRouter(
        primary_provider=groq,
        llm7_provider=llm7,
        llm7_general_model=LLM7_GENERAL_MODEL,
    )

    with pytest.raises(LLMProviderError, match="All task providers failed"):
        asyncio.run(router.chat(MESSAGES, task=TaskType.GENERAL))

    llm7.chat.assert_awaited_once_with(MESSAGES, model=LLM7_GENERAL_MODEL)
