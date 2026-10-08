from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.core.auth import configure_auth_state
from app.core.config import get_settings
from app.core.rate_limit import APIRateLimiter
from app.llm.budget import RequestBudgetPolicy
from app.llm.concurrency import ProviderConcurrencyManager
from app.llm.health import ProviderHealthManager
from app.llm.model_router import TaskModelRouter
from app.llm.registry import create_provider_registry
from app.llm.router import LLMRouter
from app.llm.task_detector import TaskDetector
from app.llm.telemetry import LLMTelemetry
from app.services.llm_service import LLMService

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_auth_state(app, settings)
    app.state.api_rate_limiter = APIRateLimiter(
        enabled=settings.ai_core_rate_limit_enabled,
        requests_per_minute=settings.ai_core_requests_per_minute,
        burst_size=settings.ai_core_burst_size,
        max_streams=settings.ai_core_max_streams,
    )
    registry = await create_provider_registry(settings)
    try:
        health_manager = ProviderHealthManager(
            enabled=settings.llm_circuit_breaker_enabled,
            failure_threshold=settings.llm_circuit_failure_threshold,
            cooldown_seconds=settings.llm_circuit_cooldown_seconds,
        )
        budget_policy = RequestBudgetPolicy(
            enabled=settings.llm_request_budget_enabled,
            standard_seconds=settings.llm_standard_budget_seconds,
            reasoning_seconds=settings.llm_reasoning_budget_seconds,
            code_seconds=settings.llm_code_budget_seconds,
            long_context_seconds=settings.llm_long_context_budget_seconds,
            multimodal_seconds=settings.llm_multimodal_budget_seconds,
        )
        telemetry = LLMTelemetry(enabled=settings.llm_telemetry_enabled)
        concurrency_manager = ProviderConcurrencyManager(
            enabled=settings.llm_provider_concurrency_enabled,
            default_limit=settings.llm_provider_default_max_concurrency,
            limits={
                "groq": settings.llm_groq_max_concurrency,
                "openrouter": settings.llm_openrouter_max_concurrency,
                "gemini": settings.llm_gemini_max_concurrency,
                "cloudflare": settings.llm_cloudflare_max_concurrency,
                "ollama": settings.llm_ollama_max_concurrency,
                "kilo": settings.llm_kilo_max_concurrency,
                "llm7": settings.llm_llm7_max_concurrency,
            },
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
        app.state.request_budget_policy = budget_policy
        app.state.llm_telemetry = telemetry
        app.state.provider_concurrency_manager = concurrency_manager
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
                budget_policy=budget_policy,
                telemetry=telemetry,
                concurrency_manager=concurrency_manager,
            ),
            model_router=TaskModelRouter(
                fast_model=settings.groq_fast_model,
                reasoning_model=settings.groq_reasoning_model,
            ),
            task_detector=TaskDetector(
                long_context_chars=settings.llm_auto_long_context_chars,
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
