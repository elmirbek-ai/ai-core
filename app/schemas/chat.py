from typing import Annotated, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

from app.llm.task import TaskType


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str = Field(min_length=1)


class ImageURL(BaseModel):
    url: HttpUrl

    @field_validator("url")
    @classmethod
    def require_https(cls, value: HttpUrl) -> HttpUrl:
        if value.scheme != "https":
            raise ValueError("image URL must use HTTPS")
        return value


class ImageURLContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Annotated[
    TextContentPart | ImageURLContentPart,
    Field(discriminator="type"),
]
StructuredContent = Annotated[list[ContentPart], Field(min_length=1)]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | StructuredContent = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    task: TaskType = TaskType.GENERAL


class ChatResponse(BaseModel):
    provider: str
    model: str
    content: str
