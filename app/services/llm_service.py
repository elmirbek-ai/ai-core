from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from app.llm.capabilities import messages_contain_images
from app.llm.exceptions import LLMProviderError
from app.llm.model_router import TaskModelRouter
from app.llm.router import LLMRouter
from app.llm.task import TaskType
from app.llm.task_detector import TaskDetector
from app.llm.streaming import StreamEvent
from app.schemas.chat import ChatMessage


class LLMService:
    def __init__(
        self,
        router: LLMRouter,
        model_router: TaskModelRouter,
        task_detector: TaskDetector | None = None,
    ) -> None:
        self.router = router
        self.model_router = model_router
        self.task_detector = task_detector or TaskDetector()

    async def chat(
        self,
        messages: list[ChatMessage],
        task: TaskType = TaskType.GENERAL,
    ) -> dict[str, str]:
        payload, resolved_task, selected_model = self._prepare_request(
            messages,
            task,
        )
        return await self.router.chat(
            payload,
            model=selected_model,
            task=resolved_task,
        )

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        task: TaskType = TaskType.GENERAL,
    ) -> AsyncIterator[StreamEvent]:
        payload, resolved_task, selected_model = self._prepare_request(
            messages,
            task,
        )
        stream = self.router.stream_chat(
            payload,
            model=selected_model,
            task=resolved_task,
        )
        async with aclosing(stream):
            async for event in stream:
                yield event

    def _prepare_request(
        self,
        messages: list[ChatMessage],
        task: TaskType,
    ) -> tuple[list[dict[str, Any]], TaskType, str | None]:
        payload = [message.model_dump(mode="json") for message in messages]
        resolved_task = (
            self.task_detector.detect(payload)
            if task == TaskType.AUTO
            else task
        )
        if (
            messages_contain_images(payload)
            and resolved_task != TaskType.MULTIMODAL
        ):
            raise LLMProviderError(
                "Image content requires a multimodal task",
            ) from None
        selected_model = self.model_router.select_model(resolved_task)
        return payload, resolved_task, selected_model
