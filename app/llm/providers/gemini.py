from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

from app.core.config import Settings, get_settings
from app.llm.base import BaseLLMProvider
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)


class GeminiProvider(BaseLLMProvider):
    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        settings = settings or get_settings()
        if not settings.gemini_api_key:
            raise ValueError("Gemini provider is disabled: API key is missing")

        self.model = settings.gemini_model
        self.client = client or AsyncOpenAI(
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
            timeout=settings.gemini_timeout_seconds,
            max_retries=settings.gemini_max_retries,
        )

    @property
    def name(self) -> str:
        return "gemini"

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
        except (AuthenticationError, PermissionDeniedError):
            raise LLMAuthenticationError(
                "Gemini authentication failed",
            ) from None
        except RateLimitError:
            raise LLMRateLimitError(
                "Gemini rate limit reached",
            ) from None
        except APITimeoutError:
            raise LLMTimeoutError(
                "Gemini request timed out",
            ) from None
        except APIConnectionError:
            raise LLMUpstreamError(
                "Gemini upstream request failed",
            ) from None
        except APIStatusError as exc:
            if exc.status_code >= 500:
                raise LLMUpstreamError(
                    "Gemini upstream request failed",
                ) from None
            raise LLMProviderError(
                "Gemini provider request failed",
            ) from None
        except OpenAIError:
            raise LLMProviderError(
                "Gemini provider request failed",
            ) from None
        except Exception:
            raise LLMProviderError(
                "Gemini provider request failed",
            ) from None

        return {
            "provider": self.name,
            "model": response.model,
            "content": response.choices[0].message.content or "",
        }

    async def close(self) -> None:
        await self.client.close()
