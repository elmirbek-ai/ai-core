import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.llm.error_mapping import map_provider_exception
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)


SECRET = "sensitive-sdk-message"
SECRET_URL = "https://provider.invalid/private-endpoint"


def api_status_error(status_code: int) -> APIStatusError:
    request = httpx2.Request("POST", SECRET_URL)
    response = httpx2.Response(status_code, request=request)
    return APIStatusError(
        SECRET,
        response=response,
        body={"detail": SECRET},
    )


@pytest.mark.parametrize(
    ("status_code", "expected_type"),
    [
        (401, LLMAuthenticationError),
        (403, LLMAuthenticationError),
        (429, LLMRateLimitError),
        (500, LLMUpstreamError),
        (502, LLMUpstreamError),
        (400, LLMProviderError),
    ],
)
def test_http_status_mapping(
    status_code: int,
    expected_type: type[LLMProviderError],
) -> None:
    mapped = map_provider_exception(
        api_status_error(status_code),
        "test-provider",
    )

    assert type(mapped) is expected_type
    assert SECRET not in str(mapped)
    assert SECRET_URL not in str(mapped)


def test_timeout_mapping() -> None:
    error = APITimeoutError(
        request=httpx2.Request("POST", SECRET_URL),
    )

    mapped = map_provider_exception(error, "test-provider")

    assert type(mapped) is LLMTimeoutError
    assert SECRET_URL not in str(mapped)


def test_connection_error_mapping() -> None:
    error = APIConnectionError(
        message=SECRET,
        request=httpx2.Request("POST", SECRET_URL),
    )

    mapped = map_provider_exception(error, "test-provider")

    assert type(mapped) is LLMUpstreamError
    assert SECRET not in str(mapped)
    assert SECRET_URL not in str(mapped)


def test_unknown_exception_mapping() -> None:
    mapped = map_provider_exception(
        RuntimeError(f"{SECRET} {SECRET_URL}"),
        "test-provider",
    )

    assert type(mapped) is LLMProviderError
    assert SECRET not in str(mapped)
    assert SECRET_URL not in str(mapped)
