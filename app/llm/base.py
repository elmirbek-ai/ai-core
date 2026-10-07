from abc import ABC, abstractmethod
from typing import Any

from app.llm.capabilities import ProviderCapabilities, TEXT_ONLY_CAPABILITIES


class BaseLLMProvider(ABC):
    @property
    def capabilities(self) -> ProviderCapabilities:
        return TEXT_ONLY_CAPABILITIES

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
