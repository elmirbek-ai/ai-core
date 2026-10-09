"""Offline admission guard, not a pricing service or adaptive routing engine.

Reviewed product-policy evidence: 2026-10-09, docs/providers/ZERO_COST_POLICY.md.
Remote account billing cannot be inspected here; conditional free modes require
operator verification before enabling them. Unknown models fail closed.
"""

import re
from enum import StrEnum
from types import MappingProxyType

from app.core.config import Settings


class ZeroCostEligibility(StrEnum):
    ELIGIBLE_FREE = "ELIGIBLE_FREE"
    CONDITIONAL_FREE = "CONDITIONAL_FREE"
    NOT_VERIFIED = "NOT_VERIFIED"
    REJECTED_PAID = "REJECTED_PAID"


PROVIDER_ELIGIBILITY = MappingProxyType(
    {
        "groq": ZeroCostEligibility.ELIGIBLE_FREE,
        "openrouter": ZeroCostEligibility.CONDITIONAL_FREE,
        "gemini": ZeroCostEligibility.ELIGIBLE_FREE,
        "cloudflare": ZeroCostEligibility.CONDITIONAL_FREE,
        "ollama": ZeroCostEligibility.NOT_VERIFIED,
        "kilo": ZeroCostEligibility.CONDITIONAL_FREE,
        "llm7": ZeroCostEligibility.CONDITIONAL_FREE,
        "openai": ZeroCostEligibility.REJECTED_PAID,
        "deepseek": ZeroCostEligibility.REJECTED_PAID,
        "mistral": ZeroCostEligibility.CONDITIONAL_FREE,
        "anthropic": ZeroCostEligibility.REJECTED_PAID,
        "xai": ZeroCostEligibility.REJECTED_PAID,
    }
)

MODEL_SETTINGS = MappingProxyType(
    {
        "groq": ("groq_fast_model", "groq_reasoning_model"),
        "openrouter": ("openrouter_model",),
        "gemini": ("gemini_model",),
        "cloudflare": ("cloudflare_model",),
        "ollama": ("ollama_model",),
        "kilo": ("kilo_general_model", "kilo_code_model", "kilo_long_context_model"),
        "llm7": ("llm7_general_model", "llm7_reasoning_model", "llm7_code_model"),
        "openai": ("openai_model",),
    }
)


def model_eligibility(provider: str, model: str | None) -> ZeroCostEligibility:
    if PROVIDER_ELIGIBILITY.get(provider) == ZeroCostEligibility.REJECTED_PAID:
        return ZeroCostEligibility.REJECTED_PAID
    if not model or "nvidia" in model.lower():
        return ZeroCostEligibility.NOT_VERIFIED
    if (
        (provider == "groq" and model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"})
        or (provider == "gemini" and model == "gemini-3.8-flash")
        or (provider == "kilo" and model == "kilo-auto/free")
        or (
            provider == "openrouter"
            and (
                model == "openrouter/free"
                or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/-]*:free", model)
            )
        )
    ):
        return ZeroCostEligibility.ELIGIBLE_FREE
    if (provider == "cloudflare" and model == "cf/openai/gpt-oss-120b") or (
        provider == "llm7" and model == "gpt-oss:20b"
    ):
        return ZeroCostEligibility.CONDITIONAL_FREE
    return ZeroCostEligibility.NOT_VERIFIED


def require_zero_cost_model(
    provider: str, model: str | None, *, free_mode_verified: bool = False
) -> None:
    eligibility = model_eligibility(provider, model)
    if eligibility == ZeroCostEligibility.ELIGIBLE_FREE or (
        eligibility == ZeroCostEligibility.CONDITIONAL_FREE and free_mode_verified
    ):
        return
    raise ValueError("Model is not eligible under the mandatory zero-cost policy")


def free_mode_verified(settings: Settings, provider: str) -> bool:
    return {
        "cloudflare": settings.cloudflare_zero_cost_verified,
        "llm7": settings.llm7_zero_cost_verified,
    }.get(provider, False)


def configuration_allowed(settings: Settings, provider: str) -> bool:
    fields = MODEL_SETTINGS.get(provider)
    if fields is None:
        return False
    try:
        for field in fields:
            require_zero_cost_model(
                provider,
                getattr(settings, field),
                free_mode_verified=free_mode_verified(settings, provider),
            )
    except ValueError:
        return False
    return True
