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


class GroqProvider(BaseLLMProvider):
    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        settings = settings or get_settings()

        self.default_model = settings.groq_fast_model
        self.client = client or AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            timeout=settings.groq_timeout_seconds,
            max_retries=settings.groq_max_retries,
        )

    @property
    def name(self) -> str:
        return "groq"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> dict[str, str]:
        selected_model = model or self.default_model
        try:
            response = await self.client.chat.completions.create(
                model=selected_model,
                messages=messages,
            )
        except AuthenticationError:
            raise LLMAuthenticationError(
                "Groq authentication failed",
            ) from None
        except RateLimitError:
            raise LLMRateLimitError(
                "Groq rate limit reached",
            ) from None
        except APITimeoutError:
            raise LLMTimeoutError(
                "Groq request timed out",
            ) from None
        except (APIConnectionError, APIStatusError, OpenAIError):
            raise LLMUpstreamError(
                "Groq upstream request failed",
            ) from None

        return {
            "provider": self.name,
            "model": response.model,
            "content": response.choices[0].message.content or "",
        }

    async def close(self) -> None:
        await self.client.close()
