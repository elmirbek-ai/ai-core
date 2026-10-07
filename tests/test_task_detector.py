import asyncio
from types import SimpleNamespace
import time
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.llm.model_router import TaskModelRouter
from app.llm.router import LLMRouter
from app.llm.task import TaskType
from app.llm.task_detector import TaskDetector
from app.llm.telemetry import LLMTelemetry
from app.schemas.chat import ChatMessage
from app.services.llm_service import LLMService


FAST_MODEL = "openai/gpt-oss-20b"
REASONING_MODEL = "openai/gpt-oss-120b"


def detect(text: str, *, threshold: int = 12_000) -> TaskType:
    return TaskDetector(threshold).detect(
        [{"role": "user", "content": text}],
    )


@pytest.mark.parametrize(
    "text",
    [
        "What is FastAPI?",
        "What is Python?",
        "FastAPI деген эмне?",
        "Кыска жооп бер: HTTP деген эмне?",
    ],
)
def test_general_fallback_is_conservative(text: str) -> None:
    assert detect(text) == TaskType.GENERAL


@pytest.mark.parametrize(
    "text",
    [
        "Fix this code:\n```python\nprint(x\n```",
        "Traceback (most recent call last): ValueError",
        "Please debug app.py",
        "Write code for a binary search",
        "Бул Python кодду анализ кылып, катасын оңдо",
        "Исправь код в handler.ts",
    ],
)
def test_code_detection(text: str) -> None:
    assert detect(text) == TaskType.CODE


@pytest.mark.parametrize(
    "text",
    [
        "Translate this into French.",
        "Переведи этот текст на английский.",
        "Бул текстти орусчага котор: Салам дүйнө",
    ],
)
def test_translate_detection_is_multilingual(text: str) -> None:
    assert detect(text) == TaskType.TRANSLATE


@pytest.mark.parametrize(
    "text",
    [
        "Summarize the following document.",
        "Сделай краткое содержание документа.",
        "Бул документти жыйынтыктап бер.",
    ],
)
def test_summarize_detection_is_multilingual(text: str) -> None:
    assert detect(text) == TaskType.SUMMARIZE


@pytest.mark.parametrize(
    "text",
    [
        "Extract all email addresses from this text.",
        "Parse fields name, date, and amount.",
        "Email жана телефон номерлерин чыгарып бер.",
    ],
)
def test_extract_detection(text: str) -> None:
    assert detect(text) == TaskType.EXTRACT


@pytest.mark.parametrize(
    "text",
    [
        "Classify this support ticket.",
        "Определи категорию сообщения.",
        "Бул кайрылууларды категорияга бөл.",
    ],
)
def test_classify_detection(text: str) -> None:
    assert detect(text) == TaskType.CLASSIFY


@pytest.mark.parametrize(
    "text",
    [
        "Find the root cause of this outage.",
        "Compare both options and explain why one is safer.",
        "Найди причину сбоя.",
        "Бул эмне үчүн жай иштеп жатат?",
    ],
)
def test_reasoning_detection_is_multilingual(text: str) -> None:
    assert detect(text) == TaskType.REASONING


def test_latest_user_message_drives_intent() -> None:
    detector = TaskDetector()
    messages = [
        {"role": "system", "content": "Fix this code and refactor it"},
        {"role": "assistant", "content": "Analyze the root cause"},
        {"role": "user", "content": "What is FastAPI?"},
    ]

    assert detector.detect(messages) == TaskType.GENERAL


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (9, TaskType.GENERAL),
        (10, TaskType.GENERAL),
        (11, TaskType.LONG_CONTEXT),
    ],
)
def test_long_context_threshold_boundary(size: int, expected: TaskType) -> None:
    assert detect("x" * size, threshold=10) == expected


def test_long_context_uses_all_message_content() -> None:
    detector = TaskDetector(long_context_chars=10)
    messages = [
        {"role": "system", "content": "12345"},
        {"role": "assistant", "content": "12345"},
        {"role": "user", "content": "x"},
    ]

    assert detector.detect(messages) == TaskType.LONG_CONTEXT


def test_long_context_has_priority_over_summarize() -> None:
    text = "Summarize this document: " + ("x" * 30)

    assert detect(text, threshold=20) == TaskType.LONG_CONTEXT


def test_code_has_priority_over_reasoning() -> None:
    text = "Analyze the root cause and fix this code: ```python\nprint(x)\n```"

    assert detect(text) == TaskType.CODE


def test_translation_language_phrase_is_detected_with_punctuation() -> None:
    assert detect("КЫРГЫЗЧАГА КОТОР: Hello!") == TaskType.TRANSLATE


def test_language_name_alone_is_not_translation() -> None:
    assert detect("Tell me about the English language") == TaskType.GENERAL


def test_multimodal_is_not_automatically_detected() -> None:
    assert detect("Describe this image") == TaskType.GENERAL


def test_config_default_and_validation() -> None:
    settings = Settings(_env_file=None, groq_api_key="test-key")

    assert settings.llm_auto_long_context_chars == 12_000
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            groq_api_key="test-key",
            llm_auto_long_context_chars=0,
        )


def service_with_mock_router() -> tuple[LLMService, AsyncMock]:
    chat = AsyncMock(
        return_value={
            "provider": "groq",
            "model": FAST_MODEL,
            "content": "ok",
        }
    )
    service = LLMService(
        router=SimpleNamespace(chat=chat),
        model_router=TaskModelRouter(FAST_MODEL, REASONING_MODEL),
        task_detector=TaskDetector(long_context_chars=100),
    )
    return service, chat


@pytest.mark.parametrize(
    ("task", "text", "expected_task", "expected_model"),
    [
        (TaskType.CODE, "What is FastAPI?", TaskType.CODE, REASONING_MODEL),
        (
            TaskType.REASONING,
            "Fix this code: ```python\nprint(x)\n```",
            TaskType.REASONING,
            REASONING_MODEL,
        ),
        (
            TaskType.TRANSLATE,
            "Fix this code: ```python\nprint(x)\n```",
            TaskType.TRANSLATE,
            FAST_MODEL,
        ),
        (
            TaskType.AUTO,
            "Fix this Python code: ```python\nprint(x\n```",
            TaskType.CODE,
            REASONING_MODEL,
        ),
        (
            TaskType.AUTO,
            "Бул текстти орусчага котор: Салам дүйнө",
            TaskType.TRANSLATE,
            FAST_MODEL,
        ),
        (
            TaskType.AUTO,
            "FastAPI деген эмне?",
            TaskType.GENERAL,
            FAST_MODEL,
        ),
    ],
)
def test_service_resolves_only_auto_tasks(
    task: TaskType,
    text: str,
    expected_task: TaskType,
    expected_model: str,
) -> None:
    service, chat = service_with_mock_router()

    asyncio.run(
        service.chat(
            [ChatMessage(role="user", content=text)],
            task=task,
        )
    )

    chat.assert_awaited_once_with(
        [{"role": "user", "content": text}],
        model=expected_model,
        task=expected_task,
    )


def test_omitted_task_preserves_general_default() -> None:
    service, chat = service_with_mock_router()
    text = "Fix this code: ```python\nprint(x)\n```"

    asyncio.run(service.chat([ChatMessage(role="user", content=text)]))

    chat.assert_awaited_once_with(
        [{"role": "user", "content": text}],
        model=FAST_MODEL,
        task=TaskType.GENERAL,
    )


def test_telemetry_records_resolved_task() -> None:
    provider = SimpleNamespace(
        name="groq",
        chat=AsyncMock(
            return_value={
                "provider": "groq",
                "model": REASONING_MODEL,
                "content": "ok",
            }
        ),
    )
    telemetry = LLMTelemetry()
    service = LLMService(
        router=LLMRouter(provider, telemetry=telemetry),
        model_router=TaskModelRouter(FAST_MODEL, REASONING_MODEL),
        task_detector=TaskDetector(),
    )

    async def exercise() -> dict:
        await service.chat(
            [ChatMessage(role="user", content="Fix this code: ```py\nx = 1\n```")],
            task=TaskType.AUTO,
        )
        return await telemetry.snapshot()

    snapshot = asyncio.run(exercise())

    assert snapshot["requests"]["last_task"] == "code"
    assert snapshot["requests"]["by_task"]["code"]["total"] == 1


def test_one_thousand_detections_are_lightweight() -> None:
    detector = TaskDetector()
    messages = [{"role": "user", "content": "Fix this code in app.py"}]

    started = time.perf_counter()
    results = [detector.detect(messages) for _ in range(1_000)]
    elapsed = time.perf_counter() - started

    assert all(result == TaskType.CODE for result in results)
    assert elapsed < 1.0
