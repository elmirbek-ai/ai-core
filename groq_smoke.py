import os
import sys

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    api_key = os.getenv("GROQ_API_KEY")
    base_url = os.getenv("GROQ_BASE_URL")
    fast_model = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
    reasoning_model = os.getenv(
        "GROQ_REASONING_MODEL",
        "openai/gpt-oss-120b",
    )

    if not api_key:
        raise RuntimeError("GROQ_API_KEY табылган жок. .env файлын текшер.")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    for task, model in (
        ("general", fast_model),
        ("reasoning", reasoning_model),
    ):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful multilingual AI assistant.",
                },
                {
                    "role": "user",
                    "content": "Кыргызча бир сүйлөм менен өзүңдү тааныштыр.",
                },
            ],
        )

        print("TASK:", task)
        print("MODEL:", model)
        print("RESPONSE:")
        print(response.choices[0].message.content)
        print()


if __name__ == "__main__":
    main()
