import httpx
import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.llm.error_mapping import (
    map_httpx_provider_exception,
    map_provider_exception,
)
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


def native_http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", SECRET_URL)
    response = httpx.Response(
        status_code,
        request=request,
        json={"detail": SECRET},
    )
    return httpx.HTTPStatusError(
        f"{SECRET} {SECRET_URL}",
        request=request,
        response=response,
    )


@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (native_http_status_error(401), LLMAuthenticationError),
        (native_http_status_error(403), LLMAuthenticationError),
        (native_http_status_error(429), LLMRateLimitError),
        (native_http_status_error(500), LLMUpstreamError),
        (native_http_status_error(502), LLMUpstreamError),
        (native_http_status_error(400), LLMProviderError),
        (
            httpx.ReadTimeout(
                SECRET,
                request=httpx.Request("POST", SECRET_URL),
            ),
            LLMTimeoutError,
        ),
        (
            httpx.ConnectError(
                SECRET,
                request=httpx.Request("POST", SECRET_URL),
            ),
            LLMUpstreamError,
        ),
        (RuntimeError(f"{SECRET} {SECRET_URL}"), LLMProviderError),
    ],
)
def test_native_httpx_error_mapping_is_sanitized(
    error: Exception,
    expected_type: type[LLMProviderError],
) -> None:
    mapped = map_httpx_provider_exception(error, "ollama")

    assert type(mapped) is expected_type
    assert SECRET not in str(mapped)
    assert SECRET_URL not in str(mapped)
