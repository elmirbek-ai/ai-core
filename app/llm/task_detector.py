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
        contents = [
            content
            for message in messages
            if isinstance((content := message.get("content")), str)
        ]
        if sum(len(content) for content in contents) > self.long_context_chars:
            return TaskType.LONG_CONTEXT

        latest_user_content = self._latest_user_content(messages)
        if latest_user_content is None:
            return TaskType.GENERAL
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
    def _latest_user_content(
        messages: Sequence[Mapping[str, object]],
    ) -> str | None:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
        return None

    @staticmethod
    def _normalize(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return " ".join(normalized.split())

