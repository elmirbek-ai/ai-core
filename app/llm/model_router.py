from app.llm.task import TaskType

FAST_TASKS = frozenset(
    {
        TaskType.GENERAL,
        TaskType.FAST,
        TaskType.SUMMARIZE,
        TaskType.TRANSLATE,
        TaskType.CLASSIFY,
        TaskType.EXTRACT,
        TaskType.AUTO,
    }
)

REASONING_TASKS = frozenset(
    {
        TaskType.REASONING,
        TaskType.CODE,
    }
)


class TaskModelRouter:
    def __init__(
        self,
        fast_model: str,
        reasoning_model: str,
    ) -> None:
        self.fast_model = fast_model
        self.reasoning_model = reasoning_model

    def select_model(self, task: TaskType) -> str | None:
        if task in FAST_TASKS:
            return self.fast_model
        if task in REASONING_TASKS:
            return self.reasoning_model
        if task in {TaskType.MULTIMODAL, TaskType.LONG_CONTEXT}:
            return None
        raise ValueError(f"Unsupported task type: {task}")
