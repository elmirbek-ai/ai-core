from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> dict[str, str]:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass
