from app.llm.model_router import TaskModelRouter
from app.llm.router import LLMRouter
from app.llm.task import TaskType
from app.llm.task_detector import TaskDetector
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
        payload = [
            message.model_dump()
            for message in messages
        ]

        resolved_task = (
            self.task_detector.detect(payload)
            if task == TaskType.AUTO
            else task
        )
        selected_model = self.model_router.select_model(resolved_task)
        return await self.router.chat(
            payload,
            model=selected_model,
            task=resolved_task,
        )
