import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
import json
import logging
from typing import NoReturn

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import LLMService


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1",
    tags=["AI"],
)


def serialize_sse(event: str, data: dict[str, str]) -> str:
    serialized = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event}\ndata: {serialized}\n\n"


def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service


def raise_provider_http_error(
    status_code: int,
    error: Exception,
) -> NoReturn:
    sanitized_error = LLMProviderError("LLM provider request failed")
    logger.error(
        "LLM chat request failed: %s",
        type(error).__name__,
        exc_info=(
            LLMProviderError,
            sanitized_error,
            error.__traceback__,
        ),
    )
    raise HTTPException(
        status_code=status_code,
        detail="LLM provider request failed",
    ) from None


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service),
) -> ChatResponse:
    try:
        result = await llm_service.chat(
            messages=request.messages,
            task=request.task,
        )
        return ChatResponse(**result)
    except LLMAuthenticationError as exc:
        raise_provider_http_error(status.HTTP_502_BAD_GATEWAY, exc)
    except LLMRateLimitError as exc:
        raise_provider_http_error(status.HTTP_503_SERVICE_UNAVAILABLE, exc)
    except LLMTimeoutError as exc:
        raise_provider_http_error(status.HTTP_504_GATEWAY_TIMEOUT, exc)
    except LLMUpstreamError as exc:
        raise_provider_http_error(status.HTTP_502_BAD_GATEWAY, exc)
    except LLMProviderError as exc:
        raise_provider_http_error(status.HTTP_502_BAD_GATEWAY, exc)
    except Exception as exc:
        raise_provider_http_error(status.HTTP_502_BAD_GATEWAY, exc)


@router.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service),
) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        try:
            stream = llm_service.stream_chat(
                messages=request.messages,
                task=request.task,
            )
            async with aclosing(stream):
                async for event in stream:
                    yield serialize_sse(event.event, event.data)
        except asyncio.CancelledError:
            raise
        except LLMTimeoutError:
            yield serialize_sse(
                "error",
                {"message": "LLM stream timed out"},
            )
        except Exception:
            yield serialize_sse(
                "error",
                {"message": "LLM provider temporarily unavailable"},
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
