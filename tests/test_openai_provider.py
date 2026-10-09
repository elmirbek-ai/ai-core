import asyncio
import json
import traceback
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)
from openai.types.responses import Response

from app.core.config import Settings
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.providers.openai import (
    OFFICIAL_BASE_URL,
    OpenAIProvider,
    as_responses_input,
)
from app.llm.streaming import ProviderStreamChunk

SECRET = "TEST_OPENAI_SECRET_DO_NOT_LOG"
MODEL = "candidate-test-model"
MESSAGES = [{"role": "user", "content": "hello"}]


def settings(**changes):
    return Settings(
        _env_file=None,
        groq_api_key="test-key",
        openai_api_key=SECRET,
        openai_model=MODEL,
        **changes,
    )


def client_for(response=None, *, error=None):
    return SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(return_value=response, side_effect=error)
        ),
        close=AsyncMock(),
    )


def completed(**changes):
    return SimpleNamespace(
        status="completed", error=None, output_text="hello", **changes
    )


def done():
    return SimpleNamespace(type="response.completed", response=completed())


class FakeStream:
    def __init__(self, events=(), *, error=None, entered=None):
        self.events = iter(events)
        self.error = error
        self.entered = entered
        self.close = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = next(self.events, None)
        if event is not None:
            return event
        if self.entered is not None:
            self.entered.set()
            await asyncio.Event().wait()
        if self.error:
            raise self.error
        raise StopAsyncIteration


def assert_secret_safe(error):
    assert SECRET not in str(error)
    assert SECRET not in "".join(traceback.format_exception(error))
    assert error.__cause__ is None


def test_name_capabilities_and_client_configuration(monkeypatch):
    client = client_for(completed())
    captured = []

    def factory(**kwargs):
        captured.append(kwargs)
        return client

    monkeypatch.setattr("app.llm.providers.openai.AsyncOpenAI", factory)
    provider = OpenAIProvider(settings(openai_timeout_seconds=11, openai_max_retries=1))
    assert provider.name == "openai"
    assert provider.capabilities.text and provider.capabilities.streaming
    assert not provider.capabilities.images
    assert captured == [
        {
            "api_key": SECRET,
            "base_url": OFFICIAL_BASE_URL,
            "timeout": 11.0,
            "max_retries": 1,
        }
    ]
    asyncio.run(provider.close())
    client.close.assert_awaited_once()


def test_default_settings_resolution_and_image_opt_in(monkeypatch):
    config = settings(openai_supports_images=True)
    client = client_for(completed())
    monkeypatch.setattr("app.llm.providers.openai.get_settings", lambda: config)
    assert OpenAIProvider(client=client).capabilities.images


def test_missing_key_rejected_before_client_creation(monkeypatch):
    config = Settings(_env_file=None, groq_api_key="test-key", openai_api_key=None)
    factory = AsyncMock()
    monkeypatch.setattr("app.llm.providers.openai.AsyncOpenAI", factory)
    with pytest.raises(ValueError, match="API key is missing"):
        OpenAIProvider(config)
    factory.assert_not_called()


def test_client_initialization_failure_is_sanitized(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr("app.llm.providers.openai.AsyncOpenAI", broken)
    with pytest.raises(LLMProviderError) as captured:
        OpenAIProvider(settings())
    assert_secret_safe(captured.value)


def test_role_order_plain_and_structured_text_conversion():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "ask"},
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"},
            ],
        },
    ]
    assert as_responses_input(messages, images=False) == [
        messages[0],
        messages[1],
        {"role": "assistant", "content": [{"type": "input_text", "text": "answer"}]},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "one"},
                {"type": "input_text", "text": "two"},
            ],
        },
    ]
    assert messages[2]["content"][0]["type"] == "text"


def image_messages(url="https://assets.invalid/image.png"):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]


def test_image_conversion_uses_https_input_image_without_fetch():
    assert as_responses_input(image_messages(), images=True) == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe"},
                {
                    "type": "input_image",
                    "image_url": "https://assets.invalid/image.png",
                    "detail": "auto",
                },
            ],
        }
    ]


@pytest.mark.parametrize(
    "url",
    [
        "http://assets.invalid/image.png",
        "file:///private/image.png",
        "C:/private/image.png",
        "data:image/png;base64,AAAA",
        "AAAA",
    ],
)
def test_invalid_image_source_is_safe_and_never_calls_sdk(url):
    client = client_for(completed())
    provider = OpenAIProvider(settings(openai_supports_images=True), client)
    with pytest.raises(LLMProviderError) as captured:
        asyncio.run(provider.chat(image_messages(url)))
    assert SECRET not in str(captured.value)
    client.responses.create.assert_not_awaited()


def test_image_disabled_and_malformed_message_never_call_sdk():
    client = client_for(completed())
    provider = OpenAIProvider(settings(), client)
    for messages in [image_messages(), [{"role": "invalid", "content": SECRET}]]:
        with pytest.raises(LLMProviderError) as captured:
            asyncio.run(provider.chat(messages))
        assert_secret_safe(captured.value)
    client.responses.create.assert_not_awaited()


def test_nonstream_responses_api_model_override_and_safe_provenance():
    response = completed(
        model=SECRET, usage=SimpleNamespace(input_tokens=100, output_tokens=20)
    )
    client = client_for(response)
    provider = OpenAIProvider(settings(), client)
    result = asyncio.run(provider.chat(MESSAGES, model="override-model"))
    assert result == {
        "provider": "openai",
        "model": "override-model",
        "content": "hello",
    }
    client.responses.create.assert_awaited_once_with(
        model="override-model", input=MESSAGES, store=False
    )
    assert provider.model == MODEL


def test_sdk_output_text_aggregate_handles_multiple_items_and_nontext_first():
    response = Response.model_construct(
        status="completed",
        error=None,
        output=[
            SimpleNamespace(type="reasoning"),
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="refusal", refusal="ignored"),
                    SimpleNamespace(type="output_text", text="one"),
                ],
            ),
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="two")],
            ),
        ],
    )
    provider = OpenAIProvider(settings(), client_for(response))
    assert asyncio.run(provider.chat(MESSAGES))["content"] == "onetwo"


def test_valid_empty_output_returns_empty_string():
    response = Response.model_construct(status="completed", error=None, output=[])
    provider = OpenAIProvider(settings(), client_for(response))
    assert asyncio.run(provider.chat(MESSAGES))["content"] == ""


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(status="completed", output_text=None),
        SimpleNamespace(status="completed", output_text=123),
        SimpleNamespace(output_text="text"),
        SimpleNamespace(status="incomplete", output_text=SECRET),
        Response.model_construct(
            status="completed",
            error=None,
            output=[SimpleNamespace(type="message", content=None)],
        ),
    ],
)
def test_malformed_or_incomplete_response_is_sanitized(response):
    provider = OpenAIProvider(settings(), client_for(response))
    with pytest.raises(LLMProviderError) as captured:
        asyncio.run(provider.chat(MESSAGES))
    assert_secret_safe(captured.value)


@pytest.mark.parametrize(
    "model", [None, "bad\nmodel", "https://private.invalid/model", "x" * 201]
)
def test_missing_or_unsafe_selected_model_never_calls_sdk(model):
    config = Settings(_env_file=None, groq_api_key="test-key", openai_api_key=SECRET)
    client = client_for(completed())
    provider = OpenAIProvider(config, client)
    with pytest.raises(LLMProviderError):
        asyncio.run(provider.chat(MESSAGES, model=model))
    client.responses.create.assert_not_awaited()


def sdk_errors():
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses?key=" + SECRET,
        headers={"Authorization": "Bearer " + SECRET},
    )
    return [
        (
            AuthenticationError(
                SECRET,
                response=httpx.Response(401, request=request),
                body={"secret": SECRET},
            ),
            LLMAuthenticationError,
        ),
        (
            RateLimitError(
                SECRET, response=httpx.Response(429, request=request), body=SECRET
            ),
            LLMRateLimitError,
        ),
        (APITimeoutError(request=request), LLMTimeoutError),
        (
            APIStatusError(
                SECRET, response=httpx.Response(503, request=request), body=SECRET
            ),
            LLMUpstreamError,
        ),
        (APIConnectionError(message=SECRET, request=request), LLMUpstreamError),
        (RuntimeError(SECRET), LLMProviderError),
        (LLMRateLimitError(SECRET), LLMRateLimitError),
    ]


@pytest.mark.parametrize(("error", "expected"), sdk_errors())
@pytest.mark.parametrize("streaming", [False, True])
def test_sdk_error_mapping_never_leaks_secret(error, expected, streaming, caplog):
    provider = OpenAIProvider(settings(), client_for(error=error))

    async def exercise():
        if streaming:
            return [chunk async for chunk in provider.stream_chat(MESSAGES)]
        return await provider.chat(MESSAGES)

    with pytest.raises(expected) as captured:
        asyncio.run(exercise())
    assert type(captured.value) is expected
    assert_secret_safe(captured.value)
    assert SECRET not in caplog.text


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("invalid_api_key", LLMAuthenticationError),
        ("rate_limit_exceeded", LLMRateLimitError),
        ("request_timeout", LLMTimeoutError),
        ("server_error", LLMUpstreamError),
        (SECRET, LLMProviderError),
    ],
)
def test_response_error_code_mapping_is_controlled(code, expected):
    response = SimpleNamespace(
        status="failed",
        error=SimpleNamespace(code=code, message=SECRET),
        output_text=SECRET,
    )
    provider = OpenAIProvider(settings(), client_for(response))
    with pytest.raises(expected) as captured:
        asyncio.run(provider.chat(MESSAGES))
    assert_secret_safe(captured.value)


def test_stream_text_only_deltas_and_lifecycle_no_duplicate_output():
    stream = FakeStream(
        [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.reasoning_summary_text.delta", delta=SECRET),
            SimpleNamespace(type="response.refusal.delta", delta=SECRET),
            SimpleNamespace(type="response.output_text.delta", delta=""),
            SimpleNamespace(type="response.output_text.delta", delta="hello"),
            SimpleNamespace(type="response.output_text.done", text="hello"),
            done(),
        ]
    )
    client = client_for(stream)
    provider = OpenAIProvider(settings(), client)

    async def exercise():
        return [chunk async for chunk in provider.stream_chat(MESSAGES)]

    assert asyncio.run(exercise()) == [ProviderStreamChunk("openai", MODEL, "hello")]
    client.responses.create.assert_awaited_once_with(
        model=MODEL, input=MESSAGES, stream=True, store=False
    )
    stream.close.assert_awaited_once()


@pytest.mark.parametrize(
    "event",
    [
        SimpleNamespace(),
        SimpleNamespace(type=123),
        SimpleNamespace(type="response.output_text.delta", delta=None),
        SimpleNamespace(type="response.completed"),
        SimpleNamespace(type="response.incomplete"),
        SimpleNamespace(
            type="response.failed",
            response=SimpleNamespace(
                error=SimpleNamespace(code=SECRET, message=SECRET)
            ),
        ),
        SimpleNamespace(type="error", code=SECRET, message=SECRET),
    ],
)
def test_malformed_or_error_stream_events_are_safe_and_close(event):
    stream = FakeStream([event])
    provider = OpenAIProvider(settings(), client_for(stream))

    async def exercise():
        return [chunk async for chunk in provider.stream_chat(MESSAGES)]

    with pytest.raises(LLMProviderError) as captured:
        asyncio.run(exercise())
    assert_secret_safe(captured.value)
    stream.close.assert_awaited_once()


def test_truncated_stream_after_delta_fails_and_closes():
    stream = FakeStream(
        [SimpleNamespace(type="response.output_text.delta", delta="partial")]
    )
    provider = OpenAIProvider(settings(), client_for(stream))

    async def exercise():
        output = []
        with pytest.raises(LLMUpstreamError):
            async for chunk in provider.stream_chat(MESSAGES):
                output.append(chunk.content)
        return output

    assert asyncio.run(exercise()) == ["partial"]
    stream.close.assert_awaited_once()


def test_stream_iteration_exception_is_sanitized_and_closes():
    stream = FakeStream(error=RuntimeError(SECRET))
    provider = OpenAIProvider(settings(), client_for(stream))

    async def exercise():
        return [chunk async for chunk in provider.stream_chat(MESSAGES)]

    with pytest.raises(LLMProviderError) as captured:
        asyncio.run(exercise())
    assert_secret_safe(captured.value)
    stream.close.assert_awaited_once()


@pytest.mark.parametrize("after_delta", [False, True])
def test_task_cancellation_cleans_up_without_mapping_cancelled_error(after_delta):
    async def exercise():
        entered = asyncio.Event()
        events = (
            [SimpleNamespace(type="response.output_text.delta", delta="first")]
            if after_delta
            else []
        )
        stream = FakeStream(events, entered=entered)
        provider = OpenAIProvider(settings(), client_for(stream))

        async def consume():
            async for _ in provider.stream_chat(MESSAGES):
                pass

        task = asyncio.create_task(consume())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        stream.close.assert_awaited_once()

    asyncio.run(exercise())


def test_generator_close_cleanup_and_cleanup_errors_do_not_leak(caplog):
    async def exercise():
        stream = FakeStream(
            [SimpleNamespace(type="response.output_text.delta", delta="first")]
        )
        stream.close.side_effect = RuntimeError(SECRET)
        provider = OpenAIProvider(settings(), client_for(stream))
        generator = provider.stream_chat(MESSAGES)
        assert (await anext(generator)).content == "first"
        await generator.aclose()
        stream.close.assert_awaited_once()

    asyncio.run(exercise())
    assert SECRET not in caplog.text


def test_client_close_error_is_sanitized():
    client = client_for()
    client.close.side_effect = RuntimeError(SECRET)
    provider = OpenAIProvider(settings(), client)
    with pytest.raises(LLMProviderError) as captured:
        asyncio.run(provider.close())
    assert_secret_safe(captured.value)


@pytest.mark.parametrize("streaming", [False, True])
def test_real_sdk_serialization_and_parsing_with_network_free_transport(streaming):
    requests = []
    response = {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "error": None,
        "model": MODEL,
        "output": [
            {
                "id": "msg_fixture",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "hello", "annotations": []}
                ],
            }
        ],
    }

    def handle(request):
        requests.append(json.loads(request.content))
        assert request.url.path == "/v1/responses"
        if streaming:
            events = [
                {"type": "response.created", "response": {"status": "in_progress"}},
                {"type": "response.output_text.delta", "delta": ""},
                {"type": "response.output_text.delta", "delta": "hello"},
                {"type": "response.completed", "response": response},
            ]
            content = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
            return httpx.Response(
                200, text=content, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=response)

    async def exercise():
        # MockTransport intercepts every request; no sockets or OpenAI endpoints.
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        sdk = AsyncOpenAI(
            api_key=SECRET,
            base_url="https://sdk-fixture.invalid/v1",
            http_client=http_client,
            max_retries=0,
        )
        provider = OpenAIProvider(settings(), sdk)
        try:
            if streaming:
                chunks = [chunk async for chunk in provider.stream_chat(MESSAGES)]
                assert [chunk.content for chunk in chunks] == ["hello"]
            else:
                assert (await provider.chat(MESSAGES))["content"] == "hello"
        finally:
            await provider.close()
        assert sdk.is_closed()

    asyncio.run(exercise())
    assert requests == [
        {
            "input": MESSAGES,
            "model": MODEL,
            "store": False,
            **({"stream": True} if streaming else {}),
        }
    ]
