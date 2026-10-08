import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.core.rate_limit import APIRateLimiter, APIRateLimitExceeded
from app.llm.base import BaseLLMProvider
from app.llm.exceptions import LLMTimeoutError
from app.llm.model_router import TaskModelRouter
from app.llm.router import LLMRouter
from app.llm.task import TaskType
from app.llm.task_detector import TaskDetector
from app.schemas.chat import ChatMessage
from app.services.llm_service import LLMService

PROPERTY_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    derandomize=True,
)


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class CountingFailureProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "shared"

    async def chat(self, messages, model=None):
        del messages, model
        self.calls += 1
        raise LLMTimeoutError("controlled timeout")

    async def close(self) -> None:
        return None


@PROPERTY_SETTINGS
@given(st.text(max_size=300))
def test_detector_is_deterministic_and_never_crashes_on_valid_unicode(
    text: str,
) -> None:
    messages = [{"role": "user", "content": text}]
    detector = TaskDetector(long_context_chars=1_000)

    first = detector.detect(messages)
    second = detector.detect(messages)

    assert isinstance(first, TaskType)
    assert second is first


@PROPERTY_SETTINGS
@given(
    case=st.sampled_from(
        [
            ("translate this", TaskType.TRANSLATE),
            ("ПЕРЕВЕДИ", TaskType.TRANSLATE),
            ("кыргызчага котор", TaskType.TRANSLATE),
            ("fix this code", TaskType.CODE),
            ("КОДДУ ОҢДО", TaskType.CODE),
            ("определи категорию", TaskType.CLASSIFY),
        ]
    ),
    whitespace=st.sampled_from([" ", "  ", "\t", "\n"]),
    punctuation=st.sampled_from(["", "!", "?", "...", ":"]),
)
def test_detector_normalization_preserves_multilingual_intent(
    case: tuple[str, TaskType],
    whitespace: str,
    punctuation: str,
) -> None:
    phrase, expected = case
    normalized_variant = whitespace.join(phrase.split())
    message = f"{whitespace}{punctuation}{normalized_variant}{punctuation}"

    assert TaskDetector().detect([{"role": "user", "content": message}]) is expected


@PROPERTY_SETTINGS
@given(
    task=st.sampled_from([task for task in TaskType if task is not TaskType.AUTO]),
    text=st.text(min_size=1, max_size=100),
)
def test_explicit_task_is_never_overridden(
    task: TaskType,
    text: str,
) -> None:
    router = SimpleNamespace(
        chat=AsyncMock(
            return_value={"provider": "test", "model": "test", "content": "ok"}
        )
    )
    service = LLMService(
        router=router,
        model_router=TaskModelRouter("fast-model", "reasoning-model"),
        task_detector=TaskDetector(long_context_chars=1),
    )

    asyncio.run(service.chat([ChatMessage(role="user", content=text)], task=task))

    assert router.chat.await_args.kwargs["task"] is task


@PROPERTY_SETTINGS
@given(
    requests_per_minute=st.integers(min_value=1, max_value=600),
    burst_size=st.integers(min_value=1, max_value=10),
    steps=st.lists(
        st.tuples(
            st.floats(
                min_value=0.0,
                max_value=5.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            st.booleans(),
        ),
        max_size=30,
    ),
)
def test_token_bucket_never_leaves_valid_bounds(
    requests_per_minute: int,
    burst_size: int,
    steps: list[tuple[float, bool]],
) -> None:
    async def exercise() -> None:
        clock = FakeClock()
        limiter = APIRateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
            clock=clock,
        )
        for elapsed, should_request in steps:
            clock.advance(elapsed)
            if should_request:
                try:
                    await limiter.acquire_request()
                except APIRateLimitExceeded as error:
                    assert error.retry_after_seconds > 0
            snapshot = await limiter.snapshot()
            available = snapshot["available_tokens"]
            assert isinstance(available, float)
            assert 0.0 <= available <= burst_size

    asyncio.run(exercise())


@PROPERTY_SETTINGS
@given(attempts=st.integers(min_value=0, max_value=100))
def test_disabled_token_bucket_always_bypasses(attempts: int) -> None:
    async def exercise() -> dict:
        limiter = APIRateLimiter(enabled=False, burst_size=1)
        for _ in range(attempts):
            await limiter.acquire_request()
        return await limiter.snapshot()

    snapshot = asyncio.run(exercise())

    assert snapshot["available_tokens"] == 1.0
    assert snapshot["allowed"] == 0
    assert snapshot["rate_limited"] == 0


@PROPERTY_SETTINGS
@given(
    budget=st.floats(
        min_value=0.01,
        max_value=120.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    elapsed_steps=st.lists(
        st.floats(
            min_value=0.0,
            max_value=20.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        max_size=20,
    ),
)
def test_remaining_budget_never_increases(
    budget: float,
    elapsed_steps: list[float],
) -> None:
    provider = CountingFailureProvider()
    clock = FakeClock(10.0)
    router = LLMRouter(provider, clock=clock)
    deadline = clock() + budget
    previous = budget

    for elapsed in elapsed_steps:
        clock.advance(elapsed)
        if clock() >= deadline:
            with pytest.raises(LLMTimeoutError):
                router._remaining_budget(deadline, TaskType.GENERAL)
            break
        remaining = router._remaining_budget(deadline, TaskType.GENERAL)
        assert remaining is not None
        assert 0 < remaining <= previous
        previous = remaining


@PROPERTY_SETTINGS
@given(task=st.sampled_from(list(TaskType)))
def test_duplicate_provider_identity_is_attempted_once(task: TaskType) -> None:
    provider = CountingFailureProvider()
    router = LLMRouter(
        provider,
        provider,
        provider,
        provider,
        provider,
        llm7_provider=provider,
    )

    with pytest.raises(LLMTimeoutError):
        asyncio.run(router.chat([{"role": "user", "content": "test"}], task=task))

    assert provider.calls == 1
