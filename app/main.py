from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.core.config import get_settings
from app.llm.model_router import TaskModelRouter
from app.llm.registry import create_provider_registry
from app.llm.router import LLMRouter
from app.services.llm_service import LLMService


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    registry = create_provider_registry(settings)
    try:
        primary_provider = registry.require_enabled(
            settings.llm_primary_provider,
        )
        fallback_provider = registry.get_enabled(
            settings.llm_fallback_provider,
        )
        gemini_provider = registry.get_enabled("gemini")
        cloudflare_provider = registry.get_enabled("cloudflare")
        app.state.provider_registry = registry
        app.state.llm_service = LLMService(
            router=LLMRouter(
                primary_provider=primary_provider,
                fallback_provider=fallback_provider,
                gemini_provider=gemini_provider,
                cloudflare_provider=cloudflare_provider,
            ),
            model_router=TaskModelRouter(
                fast_model=settings.groq_fast_model,
                reasoning_model=settings.groq_reasoning_model,
            ),
        )
        yield
    finally:
        await registry.close()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Reusable AI / LLM Gateway",
    lifespan=lifespan,
)


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


app.include_router(chat_router)
