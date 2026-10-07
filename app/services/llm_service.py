from app.llm.model_router import TaskModelRouter
from app.llm.router import LLMRouter
from app.llm.task import TaskType
from app.schemas.chat import ChatMessage


class LLMService:
    def __init__(
        self,
        router: LLMRouter,
        model_router: TaskModelRouter,
    ) -> None:
        self.router = router
        self.model_router = model_router

    async def chat(
        self,
        messages: list[ChatMessage],
        task: TaskType = TaskType.GENERAL,
    ) -> dict[str, str]:
        payload = [
            message.model_dump()
            for message in messages
        ]

        selected_model = self.model_router.select_model(task)
        return await self.router.chat(
            payload,
            model=selected_model,
            task=task,
        )
