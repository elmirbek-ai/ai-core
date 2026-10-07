from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from app.llm.capabilities import ProviderCapabilities, TEXT_ONLY_CAPABILITIES
from app.llm.exceptions import LLMProviderError
from app.llm.streaming import ProviderStreamChunk


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

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> AsyncIterator[ProviderStreamChunk]:
        del messages, model
        raise LLMProviderError("Provider does not support streaming")
        yield

    @abstractmethod
    async def close(self) -> None:
        pass
