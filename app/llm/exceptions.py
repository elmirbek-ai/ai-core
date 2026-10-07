class LLMProviderError(Exception):
    """Base error for failures reported by an LLM provider."""


class LLMAuthenticationError(LLMProviderError):
    """Provider credentials are missing, invalid, or rejected."""


class LLMRateLimitError(LLMProviderError):
    """Provider rate or quota limit was reached."""


class LLMTimeoutError(LLMProviderError):
    """Provider request exceeded its configured timeout."""


class LLMUpstreamError(LLMProviderError):
    """Provider API or network failed for another upstream reason."""
