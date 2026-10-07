import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
base_url = os.getenv("GROQ_BASE_URL")
model = os.getenv("GROQ_MODEL")

if not api_key:
    raise RuntimeError("GROQ_API_KEY табылган жок. .env файлын текшер.")

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful multilingual AI assistant.",
        },
        {
            "role": "user",
            "content": "Кыргызча 2 сүйлөм менен өзүңдү тааныштыр.",
        },
    ],
)

print("MODEL:", model)
print()
print("RESPONSE:")
print(response.choices[0].message.content)