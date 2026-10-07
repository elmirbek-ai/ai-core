from typing import Literal

from pydantic import BaseModel, Field

from app.llm.task import TaskType


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    task: TaskType = TaskType.GENERAL


class ChatResponse(BaseModel):
    provider: str
    model: str
    content: str
