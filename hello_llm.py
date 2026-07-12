import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.7,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful supply chain analyst."
        },
        {
            "role": "user",
            "content": "What are the 3 biggest challenges in retail supply chains?"
        }
    ]
)

print(response.choices[0].message.content)