import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatMessage, ChatRequest


@pytest.mark.parametrize("role", ["", "tool", "developer", "USER", None])
def test_schema_rejects_unknown_message_roles(role: object) -> None:
    with pytest.raises(ValidationError):
        ChatMessage.model_validate({"role": role, "content": "hello"})


def test_schema_rejects_empty_structured_content() -> None:
    with pytest.raises(ValidationError):
        ChatMessage.model_validate({"role": "user", "content": []})


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/image.jpg",
        "data:image/png;base64,AAAA",
        "file:///tmp/image.jpg",
        "ftp://example.com/image.jpg",
        "not-a-url",
    ],
)
def test_schema_rejects_non_https_or_embedded_image_sources(url: str) -> None:
    with pytest.raises(ValidationError):
        ChatMessage.model_validate(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        )


def test_schema_accepts_https_image_without_fetching_it() -> None:
    message = ChatMessage.model_validate(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                },
            ],
        }
    )

    dumped = message.model_dump(mode="json")
    assert dumped["content"][0]["image_url"]["url"] == ("https://example.com/image.jpg")


@pytest.mark.parametrize("task", ["unknown", "GENERAL", "", None, 1])
def test_request_schema_rejects_invalid_task_values(task: object) -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "task": task,
            }
        )
