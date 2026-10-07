from fastapi import APIRouter, HTTPException

from app.llm.groq import GroqProvider
from app.schemas.chat import ChatRequest, ChatResponse


router = APIRouter(
    prefix="/v1",
    tags=["AI"],
)

provider = GroqProvider()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        messages = [
            message.model_dump()
            for message in request.messages
        ]

        result = await provider.chat(messages)

        return ChatResponse(**result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"LLM request failed: {exc}",
        ) from exc