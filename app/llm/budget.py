from dataclasses import dataclass

from app.llm.model_router import FAST_TASKS
from app.llm.task import TaskType


@dataclass(frozen=True, slots=True)
class RequestBudgetPolicy:
    enabled: bool = True
    standard_seconds: float = 30.0
    reasoning_seconds: float = 60.0
    code_seconds: float = 60.0
    long_context_seconds: float = 90.0
    multimodal_seconds: float = 90.0

    def __post_init__(self) -> None:
        budgets = (
            self.standard_seconds,
            self.reasoning_seconds,
            self.code_seconds,
            self.long_context_seconds,
            self.multimodal_seconds,
        )
        if any(budget <= 0 for budget in budgets):
            raise ValueError("Request budgets must be greater than zero")

    def resolve(self, task: TaskType) -> float | None:
        if not self.enabled:
            return None
        if task in FAST_TASKS:
            return self.standard_seconds
        if task == TaskType.REASONING:
            return self.reasoning_seconds
        if task == TaskType.CODE:
            return self.code_seconds
        if task == TaskType.LONG_CONTEXT:
            return self.long_context_seconds
        if task == TaskType.MULTIMODAL:
            return self.multimodal_seconds
        raise ValueError(f"Unsupported task type: {task}")
