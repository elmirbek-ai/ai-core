import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.llm.capabilities import ProviderCapabilities
from app.llm.exceptions import LLMProviderError, LLMTimeoutError
from app.llm.model_router import TaskModelRouter
from app.llm.providers.cloudflare import CloudflareProvider
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.groq import GroqProvider
from app.llm.providers.kilo import KiloProvider
from app.llm.providers.llm7 import LLM7Provider
from app.llm.providers.ollama import OllamaProvider
from app.llm.providers.openrouter import OpenRouterProvider
from app.llm.router import LLMRouter
from app.llm.task import TaskType
from app.llm.task_detector import TaskDetector
from app.llm.telemetry import LLMTelemetry
from app.schemas.chat import ChatMessage, ChatRequest
from app.services.llm_service import LLMService


IMAGE_URL = "https://example.com/public-image.jpg"
MULTIMODAL_CONTENT = [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
]
MULTIMODAL_MESSAGES = [{"role": "user", "content": MULTIMODAL_CONTENT}]
GEMINI_RESULT = {
    "provider": "gemini",
    "model": "gemini-3.8-flash",
    "content": "An example image.",
}
OPENROUTER_RESULT = {
    "provider": "openrouter",
    "model": "vision/model",
    "content": "An image.",
}


def provider(
    name: str,
    *,
    images: bool = False,
    result: dict[str, str] | None = None,
    error: Exception | None = None,
):
    return SimpleNamespace(
        name=name,
        capabilities=ProviderCapabilities(images=images),
        chat=AsyncMock(return_value=result, side_effect=error),
        close=AsyncMock(),
    )


def test_string_content_remains_valid() -> None:
    request = ChatRequest.model_validate(
        {"messages": [{"role": "user", "content": "Hello"}]}
    )

    assert request.messages[0].content == "Hello"
    assert request.task == TaskType.GENERAL


def test_text_content_part_is_valid() -> None:
    message = ChatMessage.model_validate(
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    )

    assert message.model_dump(mode="json")["content"] == [
        {"type": "text", "text": "Hello"}
    ]


def test_image_url_content_part_is_valid() -> None:
    message = ChatMessage.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": IMAGE_URL}}
            ],
        }
    )

    assert message.model_dump(mode="json")["content"][0]["image_url"] == {
        "url": IMAGE_URL
    }


def test_mixed_text_and_image_content_is_valid() -> None:
    message = ChatMessage.model_validate(
        {"role": "user", "content": MULTIMODAL_CONTENT}
    )

    assert message.model_dump(mode="json")["content"] == MULTIMODAL_CONTENT


def test_unknown_content_part_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatMessage.model_validate(
            {
                "role": "user",
                "content": [{"type": "audio", "url": "https://example.com/a"}],
            }
        )


@pytest.mark.parametrize(
    "image_url",
    [
        "not-a-url",
        "http://example.com/image.jpg",
        "data:image/png;base64,AAAA",
    ],
)
def test_malformed_or_unsupported_image_url_is_rejected(image_url: str) -> None:
    with pytest.raises(ValidationError):
        ChatMessage.model_validate(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}}
                ],
            }
        )


def test_auto_with_image_detects_multimodal() -> None:
    assert TaskDetector().detect(MULTIMODAL_MESSAGES) == TaskType.MULTIMODAL


def test_auto_structured_text_preserves_existing_detection() -> None:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Fix this code in app.py"}],
        }
    ]

    assert TaskDetector().detect(messages) == TaskType.CODE


def test_image_has_priority_over_other_auto_intents() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Fix this code and translate it"},
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            ],
        }
    ]

    assert (
        TaskDetector(long_context_chars=10).detect(messages)
        == TaskType.MULTIMODAL
    )


def test_gemini_is_explicitly_image_capable() -> None:
    client = SimpleNamespace(close=AsyncMock())
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        gemini_api_key="gemini-key",
    )

    gemini = GeminiProvider(settings=settings, client=client)

    assert gemini.capabilities.images is True


@pytest.mark.parametrize(
    "provider_type",
    [
        GroqProvider,
        CloudflareProvider,
        OllamaProvider,
        KiloProvider,
        LLM7Provider,
    ],
)
def test_configured_text_providers_default_to_no_image_capability(
    provider_type,
) -> None:
    instance = object.__new__(provider_type)

    assert instance.capabilities.images is False


@pytest.mark.parametrize("supports_images", [False, True])
def test_openrouter_image_capability_is_config_driven(
    supports_images: bool,
) -> None:
    client = SimpleNamespace(close=AsyncMock())
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openrouter_api_key="openrouter-key",
        openrouter_supports_images=supports_images,
    )

    openrouter = OpenRouterProvider(settings=settings, client=client)

    assert openrouter.capabilities.images is supports_images


def test_openrouter_image_capability_defaults_to_disabled() -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openrouter_api_key="openrouter-key",
    )
    openrouter = OpenRouterProvider(
        settings=settings,
        client=SimpleNamespace(close=AsyncMock()),
    )

    assert settings.openrouter_supports_images is False
    assert openrouter.capabilities.images is False


def test_gemini_passes_structured_content_to_upstream() -> None:
    create = AsyncMock(
        return_value=SimpleNamespace(
            model="gemini-3.8-flash",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="A small image."),
                )
            ],
        )
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=AsyncMock(),
    )
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        gemini_api_key="gemini-key",
    )
    gemini = GeminiProvider(settings=settings, client=client)

    result = asyncio.run(gemini.chat(MULTIMODAL_MESSAGES))

    assert result["content"] == "A small image."
    create.assert_awaited_once_with(
        model="gemini-3.8-flash",
        messages=MULTIMODAL_MESSAGES,
    )


def test_multimodal_gemini_success_stops_chain() -> None:
    gemini = provider("gemini", images=True, result=GEMINI_RESULT)
    openrouter = provider("openrouter", images=True, result=OPENROUTER_RESULT)
    groq = provider("groq", result={})
    router = LLMRouter(groq, openrouter, gemini)

    result = asyncio.run(
        router.chat(MULTIMODAL_MESSAGES, task=TaskType.MULTIMODAL)
    )

    assert result == GEMINI_RESULT
    openrouter.chat.assert_not_awaited()
    groq.chat.assert_not_awaited()


def test_multimodal_can_fallback_to_image_capable_openrouter() -> None:
    gemini = provider(
        "gemini",
        images=True,
        error=LLMTimeoutError("private upstream detail"),
    )
    openrouter = provider("openrouter", images=True, result=OPENROUTER_RESULT)
    groq = provider("groq", result={})
    router = LLMRouter(groq, openrouter, gemini)

    result = asyncio.run(
        router.chat(MULTIMODAL_MESSAGES, task=TaskType.MULTIMODAL)
    )

    assert result == OPENROUTER_RESULT
    groq.chat.assert_not_awaited()


def test_openrouter_without_image_capability_is_skipped() -> None:
    gemini = provider(
        "gemini",
        images=True,
        error=LLMTimeoutError("private upstream detail"),
    )
    openrouter = provider("openrouter", images=False, result=OPENROUTER_RESULT)
    groq = provider("groq", result={})
    router = LLMRouter(groq, openrouter, gemini)

    with pytest.raises(LLMTimeoutError):
        asyncio.run(
            router.chat(MULTIMODAL_MESSAGES, task=TaskType.MULTIMODAL)
        )

    openrouter.chat.assert_not_awaited()
    groq.chat.assert_not_awaited()


def test_no_image_capable_provider_returns_safe_domain_error() -> None:
    groq = provider("groq", result={})
    openrouter = provider("openrouter", images=False, result={})
    router = LLMRouter(groq, openrouter, gemini_provider=None)

    with pytest.raises(LLMProviderError, match="No provider is available"):
        asyncio.run(
            router.chat(MULTIMODAL_MESSAGES, task=TaskType.MULTIMODAL)
        )

    groq.chat.assert_not_awaited()
    openrouter.chat.assert_not_awaited()


def test_all_text_only_providers_are_protected_from_image_payload() -> None:
    gemini = provider(
        "gemini",
        images=True,
        error=LLMTimeoutError("private upstream detail"),
    )
    openrouter = provider("openrouter", images=False, result={})
    groq = provider("groq", result={})
    cloudflare = provider("cloudflare", result={})
    ollama = provider("ollama", result={})
    kilo = provider("kilo", result={})
    llm7 = provider("llm7", result={})
    router = LLMRouter(
        groq,
        openrouter,
        gemini,
        cloudflare,
        ollama,
        kilo,
        llm7_provider=llm7,
    )

    with pytest.raises(LLMTimeoutError):
        asyncio.run(
            router.chat(MULTIMODAL_MESSAGES, task=TaskType.MULTIMODAL)
        )

    for text_provider in (openrouter, groq, cloudflare, ollama, kilo, llm7):
        text_provider.chat.assert_not_awaited()


def test_explicit_non_multimodal_task_with_image_is_rejected() -> None:
    router = SimpleNamespace(chat=AsyncMock())
    service = LLMService(
        router=router,
        model_router=TaskModelRouter("fast", "reasoning"),
        task_detector=TaskDetector(),
    )
    message = ChatMessage.model_validate(
        {"role": "user", "content": MULTIMODAL_CONTENT}
    )

    with pytest.raises(LLMProviderError):
        asyncio.run(service.chat([message], task=TaskType.CODE))

    router.chat.assert_not_awaited()


def test_auto_image_service_routes_as_multimodal() -> None:
    router = SimpleNamespace(
        chat=AsyncMock(
            return_value={
                "provider": "gemini",
                "model": "gemini-3.8-flash",
                "content": "image",
            }
        )
    )
    service = LLMService(
        router=router,
        model_router=TaskModelRouter("fast", "reasoning"),
        task_detector=TaskDetector(),
    )
    message = ChatMessage.model_validate(
        {"role": "user", "content": MULTIMODAL_CONTENT}
    )

    asyncio.run(service.chat([message], task=TaskType.AUTO))

    router.chat.assert_awaited_once_with(
        MULTIMODAL_MESSAGES,
        model=None,
        task=TaskType.MULTIMODAL,
    )


def test_multimodal_telemetry_stores_task_but_not_image_url() -> None:
    telemetry = LLMTelemetry()
    gemini = provider("gemini", images=True, result=GEMINI_RESULT)
    groq = provider("groq", result={})
    router = LLMRouter(groq, gemini_provider=gemini, telemetry=telemetry)

    async def exercise() -> dict:
        await router.chat(MULTIMODAL_MESSAGES, task=TaskType.MULTIMODAL)
        return await telemetry.snapshot()

    snapshot = asyncio.run(exercise())
    serialized = json.dumps(snapshot)

    assert snapshot["requests"]["last_task"] == "multimodal"
    assert IMAGE_URL not in serialized


def test_image_url_is_absent_from_router_logs(caplog) -> None:
    gemini = provider(
        "gemini",
        images=True,
        error=LLMTimeoutError("private upstream body"),
    )
    groq = provider("groq", result={})
    router = LLMRouter(groq, gemini_provider=gemini)

    with caplog.at_level(logging.WARNING, logger="app.llm.router"):
        with pytest.raises(LLMTimeoutError):
            asyncio.run(
                router.chat(MULTIMODAL_MESSAGES, task=TaskType.MULTIMODAL)
            )

    assert IMAGE_URL not in caplog.text
    assert "private upstream body" not in caplog.text
