from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.core.config import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Reusable AI / LLM Gateway",
)


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


app.include_router(chat_router)