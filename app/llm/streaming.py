from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import inspect
from typing import Any, Literal

from app.llm.error_mapping import map_provider_exception


@dataclass(frozen=True, slots=True)
class ProviderStreamChunk:
    provider: str
    model: str
    content: str


@dataclass(frozen=True, slots=True)
class StreamEvent:
    event: Literal["meta", "delta", "done"]
    data: dict[str, str]


async def stream_openai_chat(
    *,
    client: Any,
    provider_name: str,
    model: str,
    messages: list[dict[str, Any]],
) -> AsyncIterator[ProviderStreamChunk]:
    stream = None
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            if not content:
                continue
            actual_model = chunk.model or model
            yield ProviderStreamChunk(
                provider=provider_name,
                model=actual_model,
                content=content,
            )
    except Exception as exc:
        raise map_provider_exception(exc, provider_name) from None
    finally:
        close = getattr(stream, "close", None) or getattr(
            stream,
            "aclose",
            None,
        )
        if close is not None:
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass
