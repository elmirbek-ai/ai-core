from openai import AsyncOpenAI

from app.core.config import get_settings


class GroqProvider:
    def __init__(self) -> None:
        settings = get_settings()

        self.model = settings.groq_model
        self.client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )

    async def chat(self, messages: list[dict]) -> dict:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return {
            "model": response.model,
            "content": response.choices[0].message.content or "",
        }