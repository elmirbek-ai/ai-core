from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)

from app.core.config import Settings, get_settings
from app.llm.base import BaseLLMProvider
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)


class OpenRouterProvider(BaseLLMProvider):
    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        settings = settings or get_settings()
        if not settings.openrouter_api_key:
            raise ValueError("OpenRouter provider is disabled: API key is missing")

        self.model = settings.openrouter_model
        self.client = client or AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=settings.openrouter_timeout_seconds,
            max_retries=settings.openrouter_max_retries,
        )

    @property
    def name(self) -> str:
        return "openrouter"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> dict[str, str]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        except AuthenticationError:
            raise LLMAuthenticationError(
                "OpenRouter authentication failed",
            ) from None
        except RateLimitError:
            raise LLMRateLimitError(
                "OpenRouter rate limit reached",
            ) from None
        except APITimeoutError:
            raise LLMTimeoutError(
                "OpenRouter request timed out",
            ) from None
        except (APIConnectionError, APIStatusError, OpenAIError):
            raise LLMUpstreamError(
                "OpenRouter upstream request failed",
            ) from None

        return {
            "provider": self.name,
            "model": response.model,
            "content": response.choices[0].message.content or "",
        }

    async def close(self) -> None:
        await self.client.close()
