import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # 👈 loads .env file

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_with_groq(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a healthcare awareness assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        top_p=0.1,
        max_tokens=700
    )
    return response.choices[0].message.content.strip()
