from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.core.config import get_settings
from app.llm.model_router import TaskModelRouter
from app.llm.health import ProviderHealthManager
from app.llm.registry import create_provider_registry
from app.llm.router import LLMRouter
from app.services.llm_service import LLMService


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    registry = await create_provider_registry(settings)
    try:
        health_manager = ProviderHealthManager(
            enabled=settings.llm_circuit_breaker_enabled,
            failure_threshold=settings.llm_circuit_failure_threshold,
            cooldown_seconds=settings.llm_circuit_cooldown_seconds,
        )
        primary_provider = registry.require_enabled(
            settings.llm_primary_provider,
        )
        fallback_provider = registry.get_enabled(
            settings.llm_fallback_provider,
        )
        gemini_provider = registry.get_enabled("gemini")
        cloudflare_provider = registry.get_enabled("cloudflare")
        ollama_provider = registry.get_enabled("ollama")
        kilo_provider = registry.get_enabled("kilo")
        llm7_provider = registry.get_enabled("llm7")
        app.state.provider_registry = registry
        app.state.provider_health_manager = health_manager
        app.state.llm_service = LLMService(
            router=LLMRouter(
                primary_provider=primary_provider,
                fallback_provider=fallback_provider,
                gemini_provider=gemini_provider,
                cloudflare_provider=cloudflare_provider,
                ollama_provider=ollama_provider,
                kilo_provider=kilo_provider,
                kilo_general_model=settings.kilo_general_model,
                kilo_code_model=settings.kilo_code_model,
                kilo_long_context_model=settings.kilo_long_context_model,
                llm7_provider=llm7_provider,
                llm7_general_model=settings.llm7_general_model,
                llm7_reasoning_model=settings.llm7_reasoning_model,
                llm7_code_model=settings.llm7_code_model,
                health_manager=health_manager,
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
