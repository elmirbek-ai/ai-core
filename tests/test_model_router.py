import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.llm.model_router import TaskModelRouter
from app.llm.router import LLMRouter
from app.llm.task import TaskType
from app.schemas.chat import ChatMessage
from app.services.llm_service import LLMService

FAST_MODEL = "openai/gpt-oss-20b"
REASONING_MODEL = "openai/gpt-oss-120b"


@pytest.mark.parametrize(
    "task",
    [
        TaskType.GENERAL,
        TaskType.FAST,
        TaskType.SUMMARIZE,
        TaskType.TRANSLATE,
        TaskType.CLASSIFY,
        TaskType.EXTRACT,
        TaskType.AUTO,
    ],
)
def test_fast_tasks_select_20b(task: TaskType) -> None:
    router = TaskModelRouter(FAST_MODEL, REASONING_MODEL)

    assert router.select_model(task) == FAST_MODEL


@pytest.mark.parametrize(
    "task",
    [TaskType.REASONING, TaskType.CODE],
)
def test_reasoning_tasks_select_120b(task: TaskType) -> None:
    router = TaskModelRouter(FAST_MODEL, REASONING_MODEL)

    assert router.select_model(task) == REASONING_MODEL


@pytest.mark.parametrize(
    "task",
    [TaskType.MULTIMODAL, TaskType.LONG_CONTEXT],
)
def test_gemini_tasks_do_not_select_a_groq_model(task: TaskType) -> None:
    router = TaskModelRouter(FAST_MODEL, REASONING_MODEL)

    assert router.select_model(task) is None


def test_service_default_task_selects_20b() -> None:
    provider = SimpleNamespace(
        name="groq",
        chat=AsyncMock(
            return_value={
                "provider": "groq",
                "model": FAST_MODEL,
                "content": "response",
            }
        ),
    )
    service = LLMService(
        router=LLMRouter(provider),
        model_router=TaskModelRouter(FAST_MODEL, REASONING_MODEL),
    )

    asyncio.run(
        service.chat(
            messages=[ChatMessage(role="user", content="Hello")],
        )
    )

    provider.chat.assert_awaited_once_with(
        [{"role": "user", "content": "Hello"}],
        model=FAST_MODEL,
    )
