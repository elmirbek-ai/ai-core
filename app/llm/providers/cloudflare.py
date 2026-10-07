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


class CloudflareProvider(BaseLLMProvider):
    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        settings = settings or get_settings()
        if not settings.cloudflare_api_token:
            raise ValueError("Cloudflare provider is disabled: API token is missing")
        if not settings.cloudflare_account_id:
            raise ValueError("Cloudflare provider is disabled: account ID is missing")

        self.model = settings.cloudflare_model
        base_url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{settings.cloudflare_account_id}/ai/v1"
        )
        self.client = client or AsyncOpenAI(
            api_key=settings.cloudflare_api_token,
            base_url=base_url,
            timeout=settings.cloudflare_timeout_seconds,
            max_retries=settings.cloudflare_max_retries,
        )

    @property
    def name(self) -> str:
        return "cloudflare"

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
                "Cloudflare authentication failed",
            ) from None
        except RateLimitError:
            raise LLMRateLimitError(
                "Cloudflare rate limit reached",
            ) from None
        except APITimeoutError:
            raise LLMTimeoutError(
                "Cloudflare request timed out",
            ) from None
        except (APIConnectionError, APIStatusError, OpenAIError):
            raise LLMUpstreamError(
                "Cloudflare upstream request failed",
            ) from None
        except Exception:
            raise LLMProviderError(
                "Cloudflare provider request failed",
            ) from None

        return {
            "provider": self.name,
            "model": response.model,
            "content": response.choices[0].message.content or "",
        }

    async def close(self) -> None:
        await self.client.close()
