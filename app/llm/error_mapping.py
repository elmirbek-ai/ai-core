from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)


def map_provider_exception(
    error: Exception,
    provider_name: str,
) -> LLMProviderError:
    if isinstance(error, (AuthenticationError, PermissionDeniedError)):
        return LLMAuthenticationError(
            f"{provider_name} authentication failed",
        )
    if isinstance(error, RateLimitError):
        return LLMRateLimitError(
            f"{provider_name} rate limit reached",
        )
    if isinstance(error, APITimeoutError):
        return LLMTimeoutError(
            f"{provider_name} request timed out",
        )
    if isinstance(error, APIConnectionError):
        return LLMUpstreamError(
            f"{provider_name} upstream request failed",
        )
    if isinstance(error, APIStatusError):
        if error.status_code in {401, 403}:
            return LLMAuthenticationError(
                f"{provider_name} authentication failed",
            )
        if error.status_code == 429:
            return LLMRateLimitError(
                f"{provider_name} rate limit reached",
            )
        if error.status_code >= 500:
            return LLMUpstreamError(
                f"{provider_name} upstream request failed",
            )
    return LLMProviderError(
        f"{provider_name} provider request failed",
    )
