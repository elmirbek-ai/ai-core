from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
import unicodedata

from app.llm.task import TaskType


def _phrase(value: str) -> re.Pattern[str]:
    escaped = re.escape(value).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


CODE_PATTERNS = (
    re.compile(r"```"),
    _phrase("traceback (most recent call last)"),
    re.compile(
        r"(?<!\w)[\w.-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cpp|cs)(?!\w)",
        re.IGNORECASE,
    ),
    _phrase("fix this code"),
    _phrase("fix this python code"),
    _phrase("debug"),
    _phrase("bug"),
    _phrase("refactor"),
    _phrase("write code"),
    _phrase("код жаз"),
    _phrase("кодду оңдо"),
    _phrase("кодду анализ кыл"),
    _phrase("катасын оңдо"),
    _phrase("ошибку в коде"),
    _phrase("исправь код"),
)

TRANSLATE_PATTERNS = tuple(
    _phrase(value)
    for value in (
        "translate",
        "translate this",
        "translate into",
        "переведи",
        "перевод",
        "перевести на",
        "котор",
        "которуп бер",
        "кыргызчага котор",
        "орусчага котор",
        "англисчеге котор",
    )
)

SUMMARIZE_PATTERNS = tuple(
    _phrase(value)
    for value in (
        "summarize",
        "summary",
        "concise summary",
        "резюмируй",
        "кратко изложи",
        "сделай краткое содержание",
        "кыскача жаз",
        "жыйынтыктап бер",
        "кыскача түшүндүр",
        "кыскарт",
    )
)

EXTRACT_PATTERNS = tuple(
    _phrase(value)
    for value in (
        "extract",
        "extract fields",
        "extract entities",
        "parse fields",
        "выдели данные",
        "извлеки",
        "найди все email",
        "маалыматтарды бөлүп чыгар",
        "аттарын чыгарып бер",
        "email номерлерин чыгарып бер",
        "телефон номерлерин чыгарып бер",
    )
)

CLASSIFY_PATTERNS = tuple(
    _phrase(value)
    for value in (
        "classify",
        "categorize",
        "assign category",
        "классифицируй",
        "определи категорию",
        "категорияга бөл",
        "классификация кыл",
    )
)

REASONING_PATTERNS = (
    _phrase("analyze"),
    re.compile(
        r"(?<!\w)compare(?!\w).*?(?<!\w)and\s+explain\s+why(?!\w)",
        re.IGNORECASE | re.DOTALL,
    ),
    _phrase("evaluate trade-offs"),
    _phrase("evaluate tradeoffs"),
    _phrase("why does"),
    _phrase("root cause"),
    _phrase("step-by-step reasoning"),
    _phrase("step by step reasoning"),
    _phrase("проанализируй"),
    _phrase("сравни и объясни почему"),
    _phrase("почему происходит"),
    _phrase("найди причину"),
    _phrase("оцени варианты"),
    _phrase("анализ кыл"),
    _phrase("салыштырып түшүндүр"),
    _phrase("эмне үчүн"),
    _phrase("себебин тап"),
    _phrase("варианттарды баала"),
)


class TaskDetector:
    """Conservative, local task detection for explicit AUTO requests."""

    def __init__(self, long_context_chars: int = 12_000) -> None:
        if long_context_chars <= 0:
            raise ValueError("long_context_chars must be greater than zero")
        self.long_context_chars = long_context_chars

    def detect(
        self,
        messages: Sequence[Mapping[str, object]],
    ) -> TaskType:
        latest_user_message = self._latest_user_message(messages)
        if (
            latest_user_message is not None
            and self._content_has_image(latest_user_message.get("content"))
        ):
            return TaskType.MULTIMODAL

        total_text_length = sum(
            self._content_text_length(message.get("content"))
            for message in messages
        )
        if total_text_length > self.long_context_chars:
            return TaskType.LONG_CONTEXT

        if latest_user_message is None:
            return TaskType.GENERAL
        latest_user_content = self._content_text(
            latest_user_message.get("content"),
        )
        normalized = self._normalize(latest_user_content)

        rules = (
            (TaskType.CODE, CODE_PATTERNS),
            (TaskType.TRANSLATE, TRANSLATE_PATTERNS),
            (TaskType.SUMMARIZE, SUMMARIZE_PATTERNS),
            (TaskType.EXTRACT, EXTRACT_PATTERNS),
            (TaskType.CLASSIFY, CLASSIFY_PATTERNS),
            (TaskType.REASONING, REASONING_PATTERNS),
        )
        for task, patterns in rules:
            if any(pattern.search(normalized) for pattern in patterns):
                return task
        return TaskType.GENERAL

    @staticmethod
    def _latest_user_message(
        messages: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object] | None:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message
        return None

    @staticmethod
    def _content_has_image(content: object) -> bool:
        if not isinstance(content, list):
            return False
        return any(
            isinstance(part, Mapping) and part.get("type") == "image_url"
            for part in content
        )

    @staticmethod
    def _content_text(content: object) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        return " ".join(
            text
            for part in content
            if isinstance(part, Mapping)
            and part.get("type") == "text"
            and isinstance((text := part.get("text")), str)
        )

    @staticmethod
    def _content_text_length(content: object) -> int:
        if isinstance(content, str):
            return len(content)
        if not isinstance(content, list):
            return 0
        return sum(
            len(text)
            for part in content
            if isinstance(part, Mapping)
            and part.get("type") == "text"
            and isinstance((text := part.get("text")), str)
        )

    @staticmethod
    def _normalize(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return " ".join(normalized.split())
