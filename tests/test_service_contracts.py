import asyncio

import pytest

from app.llm.base import BaseLLMProvider
from app.llm.capabilities import TEXT_ONLY_CAPABILITIES
from app.llm.exceptions import LLMProviderError
from app.llm.model_router import TaskModelRouter
from app.llm.streaming import StreamEvent
from app.llm.task import TaskType
from app.llm.task_detector import TaskDetector
from app.schemas.chat import ChatMessage
from app.services.llm_service import LLMService


class IncompleteProvider(BaseLLMProvider):
    pass


class ConcreteProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.close_calls = 0

    @property
    def name(self) -> str:
        return "concrete"

    async def chat(self, messages, model=None):
        del messages, model
        return {"provider": self.name, "model": "model", "content": "ok"}

    async def close(self) -> None:
        self.close_calls += 1


class CapturingRouter:
    def __init__(self) -> None:
        self.chat_call = None
        self.stream_call = None
        self.stream_closed = False
        self.result = {"provider": "groq", "model": "model", "content": "ok"}

    async def chat(self, messages, model=None, task=TaskType.GENERAL):
        self.chat_call = {"messages": messages, "model": model, "task": task}
        return self.result

    async def stream_chat(self, messages, model=None, task=TaskType.GENERAL):
        self.stream_call = {"messages": messages, "model": model, "task": task}
        try:
            yield StreamEvent("meta", {"provider": "groq", "model": "model"})
            yield StreamEvent("delta", {"content": "ok"})
            yield StreamEvent("done", {})
        finally:
            self.stream_closed = True


def test_base_provider_contract_is_abstract() -> None:
    with pytest.raises(TypeError):
        IncompleteProvider()


def test_base_provider_defaults_and_unsupported_stream_are_explicit() -> None:
    provider = ConcreteProvider()

    async def exercise() -> None:
        assert provider.capabilities == TEXT_ONLY_CAPABILITIES
        stream = provider.stream_chat([])
        with pytest.raises(LLMProviderError, match="does not support streaming"):
            await anext(stream)
        await provider.close()

    asyncio.run(exercise())

    assert provider.close_calls == 1


def test_service_auto_result_and_selected_model_reach_router_unchanged() -> None:
    router = CapturingRouter()
    service = LLMService(
        router=router,
        model_router=TaskModelRouter("fast-model", "reasoning-model"),
        task_detector=TaskDetector(),
    )
    messages = [ChatMessage(role="user", content="Fix this code: app.py")]

    result = asyncio.run(service.chat(messages, task=TaskType.AUTO))

    assert result is router.result
    assert router.chat_call["task"] is TaskType.CODE
    assert router.chat_call["model"] == "reasoning-model"
    assert router.chat_call["messages"] == [
        {"role": "user", "content": "Fix this code: app.py"}
    ]


def test_service_stream_preserves_resolved_task_model_and_events() -> None:
    router = CapturingRouter()
    service = LLMService(
        router=router,
        model_router=TaskModelRouter("fast-model", "reasoning-model"),
    )

    async def exercise() -> list[StreamEvent]:
        return [
            event
            async for event in service.stream_chat(
                [ChatMessage(role="user", content="hello")],
                task=TaskType.GENERAL,
            )
        ]

    events = asyncio.run(exercise())

    assert [event.event for event in events] == ["meta", "delta", "done"]
    assert router.stream_call["task"] is TaskType.GENERAL
    assert router.stream_call["model"] == "fast-model"
    assert router.stream_closed


def test_service_consumer_close_closes_inner_router_stream() -> None:
    router = CapturingRouter()
    service = LLMService(
        router=router,
        model_router=TaskModelRouter("fast-model", "reasoning-model"),
    )

    async def exercise() -> StreamEvent:
        stream = service.stream_chat([ChatMessage(role="user", content="hello")])
        event = await anext(stream)
        await stream.aclose()
        return event

    event = asyncio.run(exercise())

    assert event.event == "meta"
    assert router.stream_closed


def test_service_rejects_image_with_non_multimodal_task_before_router() -> None:
    router = CapturingRouter()
    service = LLMService(
        router=router,
        model_router=TaskModelRouter("fast-model", "reasoning-model"),
    )
    message = ChatMessage.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                },
            ],
        }
    )

    with pytest.raises(LLMProviderError, match="multimodal task"):
        asyncio.run(service.chat([message], task=TaskType.CODE))

    assert router.chat_call is None
